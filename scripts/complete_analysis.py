import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
from datetime import datetime

# Create output directories if they don't exist
os.makedirs('../visualizations', exist_ok=True)
os.makedirs('../reports', exist_ok=True)

# Set plotting style
plt.style.use('seaborn')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = [12, 8]

def save_plot(name):
    """Save plot to visualizations directory"""
    plt.savefig(f'../visualizations/{name}.png', bbox_inches='tight', dpi=300)
    plt.close()

# 1. Load and Clean Data
print("Loading and cleaning data...")
df = pd.read_csv('../data/patient_data.csv')

# Convert date columns to datetime
df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])

# 2. Generate Visualizations
print("\nGenerating visualizations...")

# 2.1 Demographics
print("Creating demographic visualizations...")

# Age distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='Age', bins=30)
plt.title('Age Distribution of Patients')
save_plot('age_distribution')

# Gender distribution
plt.figure(figsize=(8, 6))
df['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Gender Distribution')
save_plot('gender_distribution')

# Regional distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Region')
plt.title('Patient Distribution by Region')
plt.xticks(rotation=45)
save_plot('regional_distribution')

# 2.2 Disease Patterns
print("Analyzing disease patterns...")

# Most common diagnoses
plt.figure(figsize=(12, 6))
diagnosis_counts = df['Diagnosis'].value_counts()
sns.barplot(x=diagnosis_counts.values, y=diagnosis_counts.index)
plt.title('Most Common Diagnoses')
save_plot('common_diagnoses')

# Treatment success rates
success_rates = df.groupby('Treatment')['TreatmentSuccess'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=success_rates.values, y=success_rates.index)
plt.title('Treatment Success Rates')
save_plot('treatment_success_rates')

# Heatmap of diagnoses by region
region_diagnosis = pd.crosstab(df['Region'], df['Diagnosis'])
plt.figure(figsize=(15, 8))
sns.heatmap(region_diagnosis, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Diagnosis Distribution by Region')
plt.xticks(rotation=45)
save_plot('diagnosis_by_region')

# 2.3 Treatment Analysis
print("Analyzing treatment outcomes...")

# Average length of stay by treatment
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Treatment', y='LengthOfStay')
plt.title('Length of Stay by Treatment Type')
plt.xticks(rotation=45)
save_plot('length_of_stay')

# Success rate by diagnosis and treatment
success_matrix = df.pivot_table(
    values='TreatmentSuccess',
    index='Diagnosis',
    columns='Treatment',
    aggfunc='mean'
)

plt.figure(figsize=(15, 8))
sns.heatmap(success_matrix, annot=True, fmt='.2f', cmap='RdYlGn')
plt.title('Treatment Success Rates by Diagnosis and Treatment Type')
plt.xticks(rotation=45)
save_plot('treatment_success_matrix')

# 3. Predictive Analytics
print("\nPerforming predictive analytics...")

# Prepare data for modeling
def prepare_features(df):
    features = df[['Age', 'InitialSeverity', 'Comorbidities', 'LengthOfStay']].copy()
    categorical_cols = ['Gender', 'Region', 'Treatment', 'InsuranceType', 'Diagnosis']
    
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(df[col])
    
    return features

# Prepare features and target
X = prepare_features(df)
y = df['TreatmentSuccess']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Generate classification report
print("\nModel Performance:")
model_report = classification_report(y_test, y_pred)
print(model_report)

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance in Predicting Treatment Success')
save_plot('feature_importance')

# 4. Export Data for Tableau
print("\nExporting data for Tableau...")

# Demographics by health condition
demographics = df.groupby(['Diagnosis', 'Age', 'Gender', 'Region']).size().reset_index(name='count')
demographics.to_csv('../data/tableau_demographics.csv', index=False)

# Treatment success rates
treatment_success = df.groupby(['Treatment', 'Diagnosis'])['TreatmentSuccess'].agg(['mean', 'count']).reset_index()
treatment_success.to_csv('../data/tableau_treatment_success.csv', index=False)

# Geographical distribution
geo_distribution = df.groupby(['Region', 'Diagnosis']).size().reset_index(name='count')
geo_distribution.to_csv('../data/tableau_geo_distribution.csv', index=False)

# 5. Generate Summary Report
print("\nGenerating summary report...")

report_content = f"""
# HealthConnect Data Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Dataset Overview
- Total Records: {len(df)}
- Date Range: {df['AdmissionDate'].min().strftime('%Y-%m-%d')} to {df['AdmissionDate'].max().strftime('%Y-%m-%d')}
- Number of Unique Patients: {df['PatientID'].nunique()}

## 2. Key Findings

### Demographics
- Age Range: {df['Age'].min()} to {df['Age'].max()} years
- Gender Distribution: {df['Gender'].value_counts(normalize=True).round(3) * 100}%
- Most Common Region: {df['Region'].mode()[0]}

### Disease Patterns
- Top 3 Diagnoses:
{df['Diagnosis'].value_counts().head(3).to_string()}

### Treatment Outcomes
- Overall Success Rate: {(df['TreatmentSuccess'].mean() * 100).round(2)}%
- Most Successful Treatment: {success_rates.index[0]} ({success_rates.iloc[0]:.2%})
- Average Length of Stay: {df['LengthOfStay'].mean():.1f} days

### Predictive Model Performance
{model_report}

## 3. Recommendations

1. Treatment Optimization
   - Focus on treatments with highest success rates
   - Consider patient characteristics when selecting treatments

2. Resource Allocation
   - Allocate resources based on regional disease patterns
   - Plan staffing based on average length of stay

3. Risk Management
   - Monitor high-risk patient profiles
   - Implement preventive measures for common conditions

4. Patient Care Improvements
   - Develop specialized programs for most common conditions
   - Consider demographic factors in treatment planning
"""

with open('../reports/analysis_report.md', 'w') as f:
    f.write(report_content)

print("\nAnalysis complete! Check the following directories for outputs:")
print("- /visualizations: All generated plots")
print("- /data: Tableau export files")
print("- /reports: Summary report")
