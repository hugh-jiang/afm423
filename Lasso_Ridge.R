if(!require(tidyverse)){install.packages("tidyverse")}
if(!require(lubridate)){install.packages("lubridate")}
if(!require(zoo)){install.packages("zoo")}
if(!require(randomForest)){install.packages("randomForest")}
if(!require(glmnet)){install.packages("glmnet")}
if(!require(dplyr)){install.packages("dplyr")}
if(!require(ggplot2)){install.packages("ggplot2")}
if(!require(tidyr)){install.packages("tidyr")}



library(tidyverse)
library(lubridate)
library(e1071)
library(zoo)
library(randomForest)
library(glmnet) # cv.glmnet for LASSO (alpha=1) and Ridge (alpha=0)
library(dplyr)
library(ggplot2)  # plotting
library(tidyr)    # reshaping


# --- 2. Load Fama-French 5-Factor Data  & HURST---
temp <- tempfile()
url  <- "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
download.file(url, temp, quiet = TRUE)

raw_ff5 <- read_csv(unz(temp, "F-F_Research_Data_5_Factors_2x3.csv"), skip = 3)

ff_data <- raw_ff5 %>%
  rename(date_raw = 1) %>%
  filter(nchar(as.character(date_raw)) == 6) %>%
  mutate(
    date   = floor_date(ymd(paste0(date_raw, "01")), "month"),
    Mkt_RF = as.numeric(`Mkt-RF`) / 100,
    SMB    = as.numeric(SMB) / 100,
    HML    = as.numeric(HML) / 100,
    RMW    = as.numeric(RMW) / 100,
    CMA    = as.numeric(CMA) / 100
  ) %>%
  select(date, Mkt_RF, SMB, HML, RMW, CMA)

compute_hurst_rs <- function(returns_vec) {
  n <- length(returns_vec)
  if (n < 36 || any(is.na(returns_vec))) return(NA_real_)
  
  # Mean-adjusted series
  r_demeaned <- returns_vec - mean(returns_vec)
  
  # Cumulative sum of mean-adjusted returns
  cum_r <- cumsum(r_demeaned)
  
  # R: range of the cumulative sum
  R <- max(cum_r) - min(cum_r)
  
  # S: standard deviation of the original return series
  S <- sd(returns_vec)
  
  if (S == 0 || R == 0) return(NA_real_)
  
  # Classical R/S Hurst estimate
  H <- log(R / S) / log(n)
  return(H)
}

load("data_ml.RData")

data_ml <- data_ml %>%
  mutate(date = floor_date(as.Date(date), "month")) %>%
  filter(date > "1999-12-31", date < "2019-01-01") %>%
  left_join(ff_data, by = "date") %>%
  arrange(stock_id, date) %>%
  group_by(stock_id) %>%
  mutate(
    # Rolling 36-month Hurst exponent applied to the RETURN series.
    # fill = NA leaves the first 35 months empty per stock.
    # NAs are handled within the rolling loop, not dropped globally here.
    Hurst = rollapplyr(R1M_Usd, width = 36, FUN = compute_hurst_rs, fill = NA)
  ) %>%
  ungroup()

# ── Basic inspection ──────────────────────────────────────────────────────────
cat("Dataset dimensions :", nrow(data_ml), "rows x", ncol(data_ml), "cols\n")
cat("Date range          :", as.character(range(data_ml$date)), "\n")
cat("Unique stocks       :", length(unique(data_ml$stock_id)), "\n")
cat("Unique months       :", length(unique(data_ml$date)), "\n")
cat("\nAll column names:\n")
print(names(data_ml))

# ── Confirm the target column ─────────────────────────────────────────────────
# R1M_Usd = 'return forward 1 month' — already a forward return, no lagging needed
cat("\nSample of target column (R1M_Usd):\n")
print(summary(data_ml$R1M_Usd))

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_COL  <- "R1M_Usd"                          # forward 1-month return (label)
ID_COLS     <- c("date", "stock_id")              # non-feature identifier columns
LABEL_COLS  <- c("R1M_Usd", "R3M_Usd",
                 "R6M_Usd", "R12M_Usd")          # all labels — exclude from features

memory_features     <- c("Mom_11M_Usd", "Hurst")
accounting_features <- c("Div_Yld", "Ebit_Bv")
market_features     <- c("Mkt_Cap_6M_Usd", "Pb", "Vol1Y_Usd")
ff_features        <- c("Mkt_RF", "SMB", "HML", "RMW", "CMA")
most_features <- setdiff(names(data_ml), c(ID_COLS, LABEL_COLS))

#FEATURE_COLS <- setdiff(names(data_ml), c(ID_COLS, LABEL_COLS))
#FEATURE_COLS <- c(memory_features, accounting_features, market_features, ff_features)
FEATURE_COLS <- unique(c(memory_features, accounting_features, market_features, ff_features, most_features))

cat("Number of features:", length(FEATURE_COLS), "\n")
cat("Features:\n")
print(FEATURE_COLS)

# ── Helper: winsorise a vector to [p_low, p_high] ────────────────────────────
winsorise <- function(x, p_low = 0.01, p_high = 0.99) {
  lo <- quantile(x, p_low,  na.rm = TRUE)
  hi <- quantile(x, p_high, na.rm = TRUE)
  pmin(pmax(x, lo), hi)
}

# ── Helper: cross-sectional standardise (within a single month's slice) ───────
cs_standardise <- function(x) {
  mu <- mean(x, na.rm = TRUE)
  sd <- sd(x,   na.rm = TRUE)
  if (is.na(sd) || sd == 0) return(x - mu)   # avoid divide-by-zero
  (x - mu) / sd
}

# ── Apply preprocessing month-by-month ───────────────────────────────────────
# R1M_Usd is already the forward 1-month return — no lead() needed.
# We only drop rows where the target itself is NA.
data_processed <- data_ml %>%
  arrange(date, stock_id) %>%
  filter(!is.na(R1M_Usd)) %>%                        # drop rows with no forward return
  group_by(date) %>%
  mutate(
    # Winsorise every feature cross-sectionally
    across(all_of(FEATURE_COLS), winsorise),
    # Standardise every feature cross-sectionally
    across(all_of(FEATURE_COLS), cs_standardise)
  ) %>%
  ungroup()

cat("\nPreprocessed dataset:\n")
cat("  Rows:", nrow(data_processed), "\n")
cat("  Feature NAs remaining:",
    sum(is.na(data_processed[, FEATURE_COLS])), "\n")

# ── Impute any remaining NAs with the cross-sectional mean (= 0 after standardising) ──
data_processed[, FEATURE_COLS][is.na(data_processed[, FEATURE_COLS])] <- 0

# ── Rolling-window configuration ──────────────────────────────────────────────
MIN_TRAIN_MONTHS <- 60   # minimum months before we start predicting
LAMBDA_RULE      <- "lambda.min"   # or "lambda.1se" — keep consistent
N_CV_FOLDS       <- 5

all_dates <- sort(unique(data_processed$date))
n_dates   <- length(all_dates)

# Pre-allocate results list (faster than rbind in a loop)
results_list        <- vector("list", n_dates - MIN_TRAIN_MONTHS)
lasso_feature_track <- vector("list", n_dates - MIN_TRAIN_MONTHS)  # tracks feature selection

cat("Starting rolling window. Predicting",
    n_dates - MIN_TRAIN_MONTHS, "months...\n")

# ── Main loop ─────────────────────────────────────────────────────────────────
for (i in seq(MIN_TRAIN_MONTHS, n_dates - 1)) {
  
  train_dates <- all_dates[1:i]          # EXPANDING window: all months up to i
  test_date   <- all_dates[i + 1]        # predict the very next month
  
  # ── Slice train / test ────────────────────────────────────────────────────
  train <- data_processed[data_processed$date %in% train_dates, ]
  test  <- data_processed[data_processed$date == test_date, ]
  
  X_train <- as.matrix(train[, FEATURE_COLS])
  y_train <- train$R1M_Usd
  X_test  <- as.matrix(test[, FEATURE_COLS])
  y_test  <- test$R1M_Usd
  
  # ── Time-series-safe CV: create fold IDs ordered chronologically ─────────
  # Split training rows into N_CV_FOLDS consecutive blocks (not random).
  # This ensures no future data leaks into any validation fold.
  n_train  <- nrow(X_train)
  fold_ids <- ceiling((seq_len(n_train) / n_train) * N_CV_FOLDS)
  fold_ids <- pmin(fold_ids, N_CV_FOLDS)   # clamp to [1, N_CV_FOLDS]
  
  # ── LASSO (alpha = 1) ─────────────────────────────────────────────────────
  set.seed(42)
  cv_lasso <- tryCatch(
    glmnet::cv.glmnet(
      X_train, y_train,
      alpha      = 1,             # L1 penalty → variable selection
      foldid     = fold_ids,      # time-ordered folds (no shuffle)
      standardize = FALSE         # already standardised cross-sectionally
    ),
    error = function(e) NULL
  )
  
  if (!is.null(cv_lasso)) {
    pred_lasso   <- as.numeric(predict(cv_lasso, X_test, s = LAMBDA_RULE))
    n_nonzero    <- sum(coef(cv_lasso, s = LAMBDA_RULE) != 0) - 1L  # excl. intercept
    lambda_lasso <- ifelse(LAMBDA_RULE == "lambda.min",
                           cv_lasso$lambda.min, cv_lasso$lambda.min)
  } else {
    pred_lasso   <- rep(NA_real_, nrow(X_test))
    n_nonzero    <- NA_integer_
    lambda_lasso <- NA_real_
  }
  
  # ── Ridge (alpha = 0) ─────────────────────────────────────────────────────
  set.seed(42)
  cv_ridge <- tryCatch(
    glmnet::cv.glmnet(
      X_train, y_train,
      alpha       = 0,            # L2 penalty → shrink all, select none
      foldid      = fold_ids,
      standardize = FALSE
    ),
    error = function(e) NULL
  )
  
  if (!is.null(cv_ridge)) {
    pred_ridge   <- as.numeric(predict(cv_ridge, X_test, s = LAMBDA_RULE))
    lambda_ridge <- ifelse(LAMBDA_RULE == "lambda.min",
                           cv_ridge$lambda.min, cv_ridge$lambda.min)
  } else {
    pred_ridge   <- rep(NA_real_, nrow(X_test))
    lambda_ridge <- NA_real_
  }
  
  # ── Store results for this month ──────────────────────────────────────────
  idx <- i - MIN_TRAIN_MONTHS + 1
  
  results_list[[idx]] <- data.frame(
    date         = test_date,
    stock_id     = test$stock_id,
    y_true       = y_test,
    y_pred_lasso = pred_lasso,
    y_pred_ridge = pred_ridge
  )
  
  lasso_feature_track[[idx]] <- data.frame(
    date          = test_date,
    n_nonzero     = n_nonzero,
    lambda_lasso  = lambda_lasso,
    lambda_ridge  = lambda_ridge
  )
  
  # Progress update every 12 months
  if (idx %% 12 == 0) cat("  Completed month", idx, "/ ",
                          n_dates - MIN_TRAIN_MONTHS, "\n")
}

cat("Rolling window complete.\n")

# ── Combine all monthly results into single data frames ───────────────────────
oos_predictions  <- do.call(rbind, results_list)
feature_tracking <- do.call(rbind, lasso_feature_track)

cat("OOS predictions shape:", nrow(oos_predictions), "rows x",
    ncol(oos_predictions), "cols\n")
cat("OOS date range:", as.character(range(oos_predictions$date)), "\n")
cat("\nNAs in predictions:\n")
cat("  LASSO:", sum(is.na(oos_predictions$y_pred_lasso)), "\n")
cat("  Ridge:", sum(is.na(oos_predictions$y_pred_ridge)), "\n")

# ── Quick sanity check: correlation between predictions and actuals ────────────
cat("\nRaw Pearson IC (all months pooled):\n")
cat("  LASSO:", round(cor(oos_predictions$y_true,
                          oos_predictions$y_pred_lasso, use="complete.obs"), 4), "\n")
cat("  Ridge:", round(cor(oos_predictions$y_true,
                          oos_predictions$y_pred_ridge, use="complete.obs"), 4), "\n")

# ── LASSO: number of non-zero features over time ──────────────────────────────
# This plot goes in your report — shows how sparsity evolves across market regimes.
plot(
  feature_tracking$date,
  feature_tracking$n_nonzero,
  type = "l", col = "steelblue", lwd = 1.5,
  xlab = "Date",
  ylab = "Non-zero LASSO coefficients",
  main = "LASSO Feature Selection Over the Rolling Window"
)
abline(h = median(feature_tracking$n_nonzero, na.rm = TRUE),
       col = "tomato", lty = 2)
legend("topright", legend = c("Non-zero count", "Median"),
       col = c("steelblue", "tomato"), lty = c(1, 2))

# ── 5A: Build quintile long-short portfolio returns ───────────────────────────
# Each month: rank stocks by predicted return, go LONG top 20%, SHORT bottom 20%.
# Equal-weight within each quintile.

build_ls_portfolio <- function(pred_df, pred_col) {
  # pred_df   : data frame with columns date, y_true, and the prediction column
  # pred_col  : string name of the prediction column (e.g. "y_pred_lasso")
  
  pred_df %>%
    filter(!is.na(.data[[pred_col]])) %>%
    group_by(date) %>%
    mutate(
      quintile = ntile(.data[[pred_col]], 5)
    ) %>%
    summarise(
      ret_long  = mean(y_true[quintile == 5], na.rm = TRUE),   # top quintile
      ret_short = mean(y_true[quintile == 1], na.rm = TRUE),   # bottom quintile
      ret_ls    = ret_long - ret_short,                         # long-short spread
      n_long    = sum(quintile == 5),
      n_short   = sum(quintile == 1),
      .groups   = "drop"
    )
}

portfolio_lasso <- build_ls_portfolio(oos_predictions, "y_pred_lasso")
portfolio_ridge <- build_ls_portfolio(oos_predictions, "y_pred_ridge")

# Vectors of monthly long-short returns — passed directly to Eval_Metrics
ls_returns_lasso <- portfolio_lasso$ret_ls
ls_returns_ridge <- portfolio_ridge$ret_ls

cat("Portfolio summary (LASSO long-short):\n")
cat("  Months:", nrow(portfolio_lasso), "\n")
cat("  Mean monthly return:", round(mean(ls_returns_lasso, na.rm=TRUE), 4), "\n")
cat("  SD monthly return:  ", round(sd(ls_returns_lasso, na.rm=TRUE), 4), "\n")

# ── 5B: Build weights matrix for compute_turnover() ──────────────────────────
# compute_turnover() in Eval_Metrics expects a T x N weights matrix.
# We build a simplified version: +1/n_long for long stocks, -1/n_short for short.

build_weights_matrix <- function(pred_df, pred_col) {
  # Wide weight matrix: rows = dates, cols = stock_ids, values = portfolio weight
  w_df <- pred_df %>%
    filter(!is.na(.data[[pred_col]])) %>%
    group_by(date) %>%
    mutate(
      quintile = ntile(.data[[pred_col]], 5),
      n_long   = sum(quintile == 5),
      n_short  = sum(quintile == 1),
      weight   = case_when(
        quintile == 5 ~  1 / n_long,
        quintile == 1 ~ -1 / n_short,
        TRUE           ~  0
      )
    ) %>%
    ungroup() %>%
    select(date, stock_id, weight)
  
  # Pivot to wide format
  w_wide <- w_df %>%
    tidyr::pivot_wider(names_from = stock_id, values_from = weight,
                       values_fill = 0) %>%
    arrange(date)
  
  as.matrix(w_wide[, -1])   # drop date column; return numeric matrix
}

weights_lasso_mat <- build_weights_matrix(oos_predictions, "y_pred_lasso")
weights_ridge_mat <- build_weights_matrix(oos_predictions, "y_pred_ridge")

cat("Weights matrix (LASSO) dimensions:", dim(weights_lasso_mat), "\n")

# ── 5C: Build asset returns matrix for compute_turnover() ─────────────────────
# Same T x N shape as the weights matrix, aligned by date and stock_id.

returns_wide <- oos_predictions %>%
  select(date, stock_id, y_true) %>%
  tidyr::pivot_wider(names_from = stock_id, values_from = y_true,
                     values_fill = 0) %>%
  arrange(date)

returns_mat <- as.matrix(returns_wide[, -1])

# Align column order to weights matrix
common_cols  <- intersect(colnames(weights_lasso_mat), colnames(returns_mat))
weights_lasso_mat <- weights_lasso_mat[, common_cols]
weights_ridge_mat <- weights_ridge_mat[, common_cols]
returns_mat       <- returns_mat[, common_cols]

cat("Returns matrix dimensions:", dim(returns_mat), "\n")

# ── 5D: Save the final training window — needed for path & CV plots ───────────
# Eval_Metrics functions evaluate_lasso_path(), select_lasso_lambda(),
# summarise_lasso_selection(), and ridge_bias_variance() all need a
# train/test split to plot on. We use the LAST rolling window for this.

last_train_dates  <- all_dates[1:(n_dates - 1)]
last_test_date    <- all_dates[n_dates]

last_train <- data_processed[data_processed$date %in% last_train_dates, ]
last_test  <- data_processed[data_processed$date == last_test_date, ]

X_train_final <- as.matrix(last_train[, FEATURE_COLS])
y_train_final <- last_train$R1M_Usd
X_test_final  <- as.matrix(last_test[, FEATURE_COLS])
y_test_final  <- last_test$R1M_Usd

# Fit the final models once (for the diagnostic plots)
set.seed(42)
fold_ids_final  <- ceiling((seq_len(nrow(X_train_final)) /
                              nrow(X_train_final)) * N_CV_FOLDS)
fold_ids_final  <- pmin(fold_ids_final, N_CV_FOLDS)

cv_lasso_final  <- glmnet::cv.glmnet(X_train_final, y_train_final,
                                     alpha = 1, foldid = fold_ids_final,
                                     standardize = FALSE)
fit_ridge_final <- glmnet::glmnet(X_train_final, y_train_final,
                                  alpha = 0, standardize = FALSE)

cat("Final-window models fit.\n")
cat("  LASSO lambda.min :", round(cv_lasso_final$lambda.min, 6), "\n")
cat("  Non-zero coefs   :",
    sum(coef(cv_lasso_final, s = "lambda.min") != 0) - 1, "\n")

# ── 5E: Save LASSO and Ridge outputs to SEPARATE .RData files ────────────────
# Each file is self-contained: it includes the shared matrices (X_train_final,
# X_test_final, y_train_final, y_test_final, returns_mat) so that either file
# can be loaded independently into Eval_Metrics.ipynb without needing the other.

# # ── LASSO outputs → lasso_outputs.RData ──────────────────────────────────────
# save(
#   # Predictions (Parts A, G)
#   oos_predictions,      # columns used: date, stock_id, y_true, y_pred_lasso
#   
#   # Portfolio (Part H)
#   ls_returns_lasso,     # → compute_sharpe(), compute_max_drawdown()
#   weights_lasso_mat,    # → compute_turnover()
#   returns_mat,          # shared — needed by compute_turnover()
#   
#   # LASSO diagnostics (Part C)
#   cv_lasso_final,       # → select_lasso_lambda(), summarise_lasso_selection()
#   X_train_final,        # → evaluate_lasso_path()
#   y_train_final,
#   X_test_final,
#   y_test_final,
#   FEATURE_COLS,         # → summarise_lasso_selection() needs feature names
#   
#   # Feature selection over time (your report figure)
#   feature_tracking,
#   
#   file = "lasso_outputs.RData"
# )
# cat("LASSO outputs saved to lasso_outputs.RData\n")
# 
# # ── Ridge outputs → ridge_outputs.RData ──────────────────────────────────────
# save(
#   # Predictions (Parts A, G)
#   oos_predictions,      # columns used: date, permno, y_true, y_pred_ridge
#   
#   # Portfolio (Part H)
#   ls_returns_ridge,     # → compute_sharpe(), compute_max_drawdown()
#   weights_ridge_mat,    # → compute_turnover()
#   returns_mat,          # shared — needed by compute_turnover()
#   
#   # Ridge diagnostics (Part D)
#   fit_ridge_final,      # → ridge_bias_variance()
#   X_train_final,        # shared — kept for symmetry / completeness
#   y_train_final,
#   X_test_final,
#   y_test_final,
#   FEATURE_COLS,
#   
#   file = "ridge_outputs.RData"
# )
# cat("Ridge  outputs saved to ridge_outputs.RData\n")
# cat("\nLoad in Eval_Metrics.ipynb with:\n")
# cat("  load('lasso_outputs.RData')   # for LASSO evaluation\n")
# cat("  load('ridge_outputs.RData')   # for Ridge evaluation\n")

#load("lasso_outputs.RData")
#load("ridge_outputs.RData")

save(
 # Predictions (Parts A, G)
 oos_predictions,
 file = "prediction_outputs_wFeatureAndFF5.RData"
)

# load("prediction_outputs_wFeature.RData")
# load("prediction_outputs_old.RData")
