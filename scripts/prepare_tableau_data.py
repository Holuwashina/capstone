import pandas as pd
import os

def prepare_demographics_data(df):
    """Prepare data for demographics dashboard"""
    # Age group analysis
    df['AgeGroup'] = pd.cut(df['Age'], 
                           bins=[0, 20, 30, 40, 50, 60, 70, 80, 90],
                           labels=['0-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81+'])
    
    demographics = df.groupby(['AgeGroup', 'Gender', 'Region']).size().reset_index(name='PatientCount')
    return demographics

def prepare_diagnosis_data(df):
    """Prepare data for diagnosis patterns dashboard"""
    # Add month and year for temporal analysis
    df['Month'] = pd.to_datetime(df['AdmissionDate']).dt.month
    df['Year'] = pd.to_datetime(df['AdmissionDate']).dt.year
    
    diagnosis_patterns = df.groupby(['Region', 'Diagnosis', 'Month', 'Year']).agg({
        'PatientID': 'count',
        'TreatmentSuccess': 'mean',
        'InitialSeverity': 'mean'
    }).reset_index()
    
    diagnosis_patterns.columns = ['Region', 'Diagnosis', 'Month', 'Year', 
                                'PatientCount', 'SuccessRate', 'AvgSeverity']
    return diagnosis_patterns

def prepare_treatment_data(df):
    """Prepare data for treatment outcomes dashboard"""
    treatment_outcomes = df.groupby(['Treatment', 'Diagnosis']).agg({
        'PatientID': 'count',
        'TreatmentSuccess': 'mean',
        'LengthOfStay': 'mean',
        'FollowUpVisits': 'mean'
    }).reset_index()
    
    treatment_outcomes.columns = ['Treatment', 'Diagnosis', 'PatientCount',
                                'SuccessRate', 'AvgLengthOfStay', 'AvgFollowUps']
    return treatment_outcomes

def prepare_predictive_data(df):
    """Prepare data for predictive analytics dashboard"""
    # Calculate risk scores based on various factors
    df['RiskScore'] = (
        df['InitialSeverity'] * 0.4 +
        df['Comorbidities'] * 0.3 +
        df['Age'].apply(lambda x: min((x/85) * 0.3, 0.3))
    )
    
    predictive_data = df.groupby(['Region', 'Diagnosis', 'Treatment']).agg({
        'RiskScore': ['mean', 'std'],
        'TreatmentSuccess': 'mean',
        'PatientID': 'count'
    }).reset_index()
    
    predictive_data.columns = ['Region', 'Diagnosis', 'Treatment',
                             'AvgRiskScore', 'RiskScoreStd', 'SuccessRate', 'PatientCount']
    return predictive_data

def main():
    # Read the patient data
    input_path = os.path.join('data', 'patient_data.csv')
    df = pd.read_csv(input_path)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('visualizations', 'tableau_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare and save different views of the data
    datasets = {
        'demographics': prepare_demographics_data,
        'diagnosis_patterns': prepare_diagnosis_data,
        'treatment_outcomes': prepare_treatment_data,
        'predictive_analytics': prepare_predictive_data
    }
    
    for name, prepare_func in datasets.items():
        output_data = prepare_func(df)
        output_path = os.path.join(output_dir, f'{name}.csv')
        output_data.to_csv(output_path, index=False)
        print(f"Generated {name} dataset with {len(output_data)} records")
        print(f"Saved to: {output_path}\n")

if __name__ == "__main__":
    main()
