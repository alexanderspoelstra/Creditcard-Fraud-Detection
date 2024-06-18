import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.ensemble import IsolationForest
from custom_isolation_forest import CustomIsolationForest
from hyper_parameter_tuning import tune_hyperparameters
from utils import *

# Load environment variables from .env file
FILEPATH = 'C:/Users/UALESP/Documents/Projects - local/Student Competition 2024/input_data/vlk_fraud_no_solution.csv'
OUTPUT_DIR='C:/Users/UALESP/Documents/Projects - local/Student Competition 2024/results'
PARAMETERS_FILEPATH='C:/Users/UALESP/Documents/Projects - local/Student Competition 2024/parameters/hyperparameters.csv'
HYPER_TUNING='false'

# Load data
df = load_data(filepath = FILEPATH)

# Process data
X = process_data(df,['Transaction_ID'])

# Set up model with or without parameter tuning
if HYPER_TUNING == 'true':
    # Instantiate Isolation Forest model
    model = IsolationForest()

    # Perform hyperparameter tuning
    best_params = tune_hyperparameters(model, X)

    # Instantiate CustomIsolationForest with the best hyperparameters
    custom_model = CustomIsolationForest(**best_params)

else:
    # Instantiate Isolation Forest model with default values
    custom_model = CustomIsolationForest(n_estimators=50, contamination=0.001, max_samples=0.8, max_features = 0.8, random_state=42)

# save_model(custom_model, 'C:/Users/Alexander/Downloads/VKL/model_vkl_fraud.pkl')

# Fit and predict with the model
custom_model.fit(X)
y_hat = custom_model.predict(X)
scores = custom_model.decision_function(X)

# Add predictions and scores to input data and export
results = custom_model.add_results(df,y_hat,scores)
export_results_to_csv(results, custom_model.get_params(), output_dir=OUTPUT_DIR)
