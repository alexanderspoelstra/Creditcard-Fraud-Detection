import os
from dotenv import load_dotenv

from sklearn.model_selection import GridSearchCV
import pandas as pd

# Load environment variables from .env file
load_dotenv()

def scorer(estimator,X): 
    """
    Custom scoring function for evaluating the performance of an Isolation Forest model.

    Parameters:
    - estimator: The Isolation Forest model to be evaluated.
    - X: The input data used for evaluation.

    Returns:
    - score: The score computed based on the decision function of the Isolation Forest model.
    """

    # Define Isolation Forest model
    score = estimator.fit(X).decision_function(X)
    return -score[score < 0].mean()

def tune_hyperparameters(model, X):
    """
    Tune hyperparameters of an Isolation Forest model using GridSearchCV.

    Parameters:
    - model: The Isolation Forest model to be tuned.
    - X: The input data used for tuning.

    Returns:
    - best_params: The best hyperparameters found by GridSearchCV.
    """
    
    # Parameter grid for GridSearchCV
    param_grid = {
    'n_estimators': [50, 100],
    'max_samples': [0.25, 0.8],
    'max_features': [0.8],
    'contamination': [0.001]
    }
    
    # Create GridSearchCV object with custom scorer
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scorer, verbose = 3.5)
    grid_search.fit(X)
    
    # Extract best hyperparameters found
    best_params = grid_search.best_params_
        
    # Read existing results from CSV file
    try:
        results_df = pd.read_csv(os.getenv("PARAMETERS_FILEPATH"))
    except FileNotFoundError:
        results_df = pd.DataFrame(columns=list(param_grid.keys()) + ['score'])
    
    # Append new results to DataFrame
    best_params['score'] = grid_search.best_score_
    results_df = pd.concat([results_df, pd.DataFrame([best_params])], ignore_index=True)

    # Write updated DataFrame to CSV file
    results_df.to_csv(os.getenv("PARAMETERS_FILEPATH"), index=False)
    
    # delete score
    del best_params['score']

    return best_params