<b> Please note that data_ml.RData should be placed in the root folder in order to run all code. Ipynb files contain r code. </b>

### File Organization
- `evaluate_all_models.ipynb`: contains code for comparisons and evaluating portfolio strategies based on all models
- `metrics.ipynb`: notebook of common algorithm metrics, used in the previous file
- `Lasso_Ridge.R`, `neural_network.ipynb`, `random_forest.ipynb`, `svr.ipynb`: code for each model
- `preds`: folder of final model predictions, used in `evaluate_all_models.ipynb`


### Set Up
To run ipynb notebooks with a r environment:

For a Mac
1. run in default terminal:<br>
`R`                           
`install.packages('IRkernel')`   
`IRkernel::installspec()`<br>

2. Restart VScode
3. open an ipynb, click select kernel (in the top right corner) -> Jupyter kernel -> R
