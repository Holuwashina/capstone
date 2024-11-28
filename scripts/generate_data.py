import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Define constants
NUM_PATIENTS = 1000
REGIONS = ['North', 'South', 'East', 'West', 'Central']
DIAGNOSES = [
    'Hypertension', 'Type 2 Diabetes', 'Asthma', 'Arthritis', 
    'Depression', 'Anxiety', 'Obesity', 'COPD', 
    'Heart Disease', 'Lower Back Pain'
]
TREATMENTS = [
    'Medication', 'Physical Therapy', 'Surgery', 
    'Lifestyle Changes', 'Counseling', 'Combined Therapy'
]

def generate_patient_data():
    data = []
    
    for _ in range(NUM_PATIENTS):
        age = random.randint(18, 85)
        gender = random.choice(['M', 'F'])
        region = random.choice(REGIONS)
        
        # Generate 1-3 diagnoses per patient
        num_diagnoses = random.randint(1, 3)
        diagnoses = random.sample(DIAGNOSES, num_diagnoses)
        
        for diagnosis in diagnoses:
            treatment = random.choice(TREATMENTS)
            
            # Generate realistic success rate based on various factors
            base_success = random.uniform(0.5, 0.9)
            age_factor = 1 - (age - 18) / (85 - 18) * 0.2  # Older patients might have lower success rates
            treatment_factor = 1.1 if treatment in ['Combined Therapy', 'Surgery'] else 1.0
            
            success_rate = min(1.0, base_success * age_factor * treatment_factor)
            
            # Generate admission and discharge dates
            admission_date = fake.date_between(start_date='-2y', end_date='today')
            length_of_stay = random.randint(1, 30)
            discharge_date = admission_date + timedelta(days=length_of_stay)
            
            data.append({
                'PatientID': fake.unique.random_number(digits=6),
                'Age': age,
                'Gender': gender,
                'Region': region,
                'Diagnosis': diagnosis,
                'Treatment': treatment,
                'AdmissionDate': admission_date,
                'DischargeDate': discharge_date,
                'LengthOfStay': length_of_stay,
                'TreatmentSuccess': random.random() < success_rate,
                'InitialSeverity': random.randint(1, 10),
                'Comorbidities': random.randint(0, 3),
                'InsuranceType': random.choice(['Private', 'Public', 'None']),
                'FollowUpVisits': random.randint(0, 5)
            })
    
    return pd.DataFrame(data)

def main():
    # Generate the dataset
    df = generate_patient_data()
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'patient_data.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} patient records and saved to {output_path}")
    
    # Print basic statistics
    print("\nDataset Overview:")
    print(f"Total number of records: {len(df)}")
    print(f"Unique patients: {df['PatientID'].nunique()}")
    print(f"Date range: {df['AdmissionDate'].min()} to {df['AdmissionDate'].max()}")
    print("\nSuccess rates by treatment:")
    print(df.groupby('Treatment')['TreatmentSuccess'].mean().round(3))

if __name__ == "__main__":
    main()
