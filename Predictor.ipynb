import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Load your stock data (assuming it's in a CSV file)
data = pd.read_csv('D:/data.csv')

# Assuming your dataset has a 'Date' column
# You might need to preprocess the data depending on your dataset
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Convert numeric columns to float after removing commas
numeric_columns = ['Open', 'High', 'Low', 'Close']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')

# Define features (X) and target variable (y)
X = data[['Open', 'High', 'Low', 'Close']]
y = data['Close']

# Create a column transformer for preprocessing
preprocessing = ColumnTransformer(
    transformers=[
        ('numeric', SimpleImputer(strategy='mean'), ['Open', 'High', 'Low', 'Close']),
    ],
    remainder='passthrough'  # Ignore other columns (e.g., 'Date')
)

# Create a pipeline for preprocessing and modeling
pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', HistGradientBoostingRegressor(random_state=42)),
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for grid search
param_grid = {
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [None, 5, 10],
    'model__max_iter': [100, 200, 300],
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
predictions_test = best_model.predict(X_test)

# Evaluate the best model on the test set
mse_best_model = mean_squared_error(y_test, predictions_test)
print(f'Best Model Mean Squared Error on Test Set: {mse_best_model}')

# Make predictions for the next few days (similar to the previous script)
future_days = 5
last_date = data.index[-1]
next_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='B')

# Create a DataFrame with the predicted values (excluding 'Date')
next_data = pd.DataFrame(index=next_dates, columns=['Open', 'High', 'Low', 'Close'])
next_data.index.name = 'Date'
next_data.reset_index(inplace=True)

# Make predictions using the best model
predictions_next_days_best_model = best_model.predict(next_data)

# Print the predictions for the next few days using the best model
print("\nPredictions for the Next Few Days (Best Model):")
for date, prediction in zip(next_dates, predictions_next_days_best_model):
    print(f'Date: {date.date()}, Prediction: {prediction:.2f}')
