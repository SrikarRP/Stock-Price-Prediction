# Stock-Price-Prediction
Stock Price Prediction with Hyperparameter Tuning using HistGradientBoostingRegressor

Imports:
pandas: Library for data manipulation and analysis.
enable_hist_gradient_boosting: Enable experimental histogram-based gradient boosting in scikit-learn.
HistGradientBoostingRegressor: Implementation of histogram-based gradient boosting regressor.
SimpleImputer: Imputation transformer for handling missing values.
ColumnTransformer: Applies transformers to columns of an array or pandas DataFrame.
Pipeline: Chains multiple processing steps into a single scikit-learn estimator.
train_test_split: Function to split the data into random train and test sets.
GridSearchCV: Performs grid search with cross-validation to find the best hyperparameters.
mean_squared_error: Metric to evaluate the performance of regression models.

Data Loading and Preprocessing:
Load stock data from a CSV file into a pandas DataFrame.
Convert the 'Date' column to datetime format and set it as the index.
Convert numeric columns to float after removing commas.

Feature and Target Variable Definition:
Define the features (X) as the 'Open', 'High', 'Low', and 'Close' columns.
Define the target variable (y) as the 'Close' column.

Column Transformer for Preprocessing:
Create a ColumnTransformer to apply the SimpleImputer for imputing missing values to the numeric columns.
Use 'passthrough' to ignore other columns (e.g., 'Date') during preprocessing.

Pipeline Creation:
Create a Pipeline that consists of preprocessing steps and the HistGradientBoostingRegressor model.

Train-Test Split:
Split the data into training and testing sets using 80% for training and 20% for testing.

Hyperparameter Grid:
Define a grid of hyperparameter values for the HistGradientBoostingRegressor.

Grid Search for Hyperparameter Tuning:
Use GridSearchCV to perform a grid search with cross-validation to find the best hyperparameters.

Model Evaluation:
Make predictions on the test set using the best model and calculate the mean squared error for evaluation.

Prepare Next Data for Predictions:
Prepare a DataFrame for the next few days to make predictions.

Predictions for the Next Few Days:
Use the best model to make predictions for the next few days and print the results.
Adjustments and further experiments can be made based on your specific use case and dataset.
