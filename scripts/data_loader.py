# Import required libraries
import pandas as pd
import os

def load_patient_data():
    """Load and validate patient data"""
    # Get the absolute path to the data file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'patient_data.csv')
    
    # Load the data
    df = pd.read_csv(data_path)
    return df

if __name__ == "__main__":
    # Test the data loading
    df = load_patient_data()
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nColumns:", df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
