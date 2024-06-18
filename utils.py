import pandas as pd
from datetime import datetime
import os
import joblib

def load_data(filepath):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    - filepath (str): The filepath of the CSV file containing the data to be loaded.

    Returns:
    - df (pandas DataFrame): The DataFrame containing the loaded data.
    """ 

    df = pd.read_csv(filepath)
    return df

def process_data(df, column_drop ):
    """
    Process the input data by dropping specified columns.

    Parameters:
    - df (DataFrame): The input DataFrame containing the data to be processed.
    - column_drop (list or str): The column(s) to be dropped from the input DataFrame. 
      
    Returns:
    - X (DataFrame): The processed DataFrame with specified columns dropped.
    """ 

    X = df.drop(column_drop,axis= 1)
    return X

def save_model(model, filename):
    """
    Save the Isolation Forest model to a file using joblib.
    
    Parameters:
    - model: The Isolation Forest model to be saved.
    - filename: The filename (including path) where the model will be saved.
    """
    try:
        joblib.dump(model, filename)
        print(f"Isolation Forest model saved successfully to {filename}")
    except Exception as e:
        print(f"Error occurred while saving the Isolation Forest model: {e}")

def load_model(filepath):
    """
    Load an Isolation Forest model from a file using joblib.
    
    Parameters:
    - filename: The filename (including path) from which the model will be loaded.
    
    Returns:
    - model: The loaded Isolation Forest model.
    """
    try:
        model = joblib.load(filepath)
        print(f"Isolation Forest model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"Error occurred while loading the Isolation Forest model: {e}")
        return None

def export_results_to_csv(results_df, params, output_dir):
    """
    Export a pandas DataFrame containing results to a CSV file with a filename
    constructed based on the current date, time, and parameters provided.

    Parameters:
    - results_df (DataFrame): The DataFrame containing the results to be exported to CSV.
    - params (dict): A dictionary containing parameters used in the model. 
    - output_dir (str): The directory where the CSV file will be saved. 
    """
    
    #Transform data to correct format
    results_df = results_df[['Transaction_ID','scores','Yhat']].sort_values('scores',ascending=True).head(100)
    results_df['Rank'] = results_df['scores'].rank(method = 'first')
    results_df.drop(['Yhat','scores'], axis=1,inplace=True)
    print(results_df)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get current date and time
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Construct filename with date, time, and parameters
    filename = f"results_{current_datetime}_"
    for key, value in params.items():
        filename += f"{key}_{value}_"
    filename += ".csv"
    
    # Export results DataFrame to CSV file
    results_df.to_csv(os.path.join(output_dir, filename), index=False)

