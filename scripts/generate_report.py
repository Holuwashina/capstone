import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load and prepare the patient data"""
    df = pd.read_csv(os.path.join('data', 'patient_data.csv'))
    df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
    df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])
    return df

def generate_demographic_insights(df):
    """Generate demographic analysis insights"""
    insights = []
    
    # Age distribution
    avg_age = df['Age'].mean()
    age_std = df['Age'].std()
    insights.append(f"- Average patient age is {avg_age:.1f} years (SD: {age_std:.1f})")
    
    # Gender distribution
    gender_dist = df['Gender'].value_counts(normalize=True) * 100
    insights.append(f"- Gender distribution: {gender_dist['M']:.1f}% Male, {gender_dist['F']:.1f}% Female")
    
    # Regional distribution
    region_dist = df['Region'].value_counts()
    max_region = region_dist.index[0]
    insights.append(f"- Highest patient concentration in {max_region} region ({region_dist[max_region]} patients)")
    
    return "\n".join(insights)

def generate_diagnosis_insights(df):
    """Generate diagnosis pattern insights"""
    insights = []
    
    # Most common diagnoses
    top_diagnoses = df['Diagnosis'].value_counts().head(3)
    insights.append("Top 3 diagnoses:")
    for diagnosis, count in top_diagnoses.items():
        insights.append(f"  - {diagnosis}: {count} cases")
    
    # Regional patterns
    region_diagnosis = pd.crosstab(df['Region'], df['Diagnosis'])
    for region in df['Region'].unique():
        top_diagnosis = region_diagnosis.loc[region].idxmax()
        count = region_diagnosis.loc[region][top_diagnosis]
        insights.append(f"- {region} region: Most common diagnosis is {top_diagnosis} ({count} cases)")
    
    return "\n".join(insights)

def generate_treatment_insights(df):
    """Generate treatment outcome insights"""
    insights = []
    
    # Overall success rate
    success_rate = df['TreatmentSuccess'].mean() * 100
    insights.append(f"- Overall treatment success rate: {success_rate:.1f}%")
    
    # Success by treatment type
    treatment_success = df.groupby('Treatment')['TreatmentSuccess'].agg(['mean', 'count'])
    treatment_success['mean'] = treatment_success['mean'] * 100
    
    insights.append("\nTreatment success rates:")
    for treatment in treatment_success.index:
        mean = treatment_success.loc[treatment, 'mean']
        count = treatment_success.loc[treatment, 'count']
        insights.append(f"  - {treatment}: {mean:.1f}% success rate ({count} cases)")
    
    return "\n".join(insights)

def generate_predictive_insights(df):
    """Generate predictive analytics insights"""
    insights = []
    
    # Create a basic risk score
    df['RiskScore'] = df['InitialSeverity'] * 0.4 + df['Comorbidities'] * 0.3 + df['Age'].apply(lambda x: min((x/85) * 0.3, 0.3))
    
    # Risk factors analysis
    high_risk = df[df['RiskScore'] > df['RiskScore'].quantile(0.75)]
    
    insights.append("Risk Factor Analysis:")
    insights.append(f"- {len(high_risk)} patients identified as high-risk")
    insights.append(f"- Average age in high-risk group: {high_risk['Age'].mean():.1f} years")
    insights.append(f"- Most common diagnosis in high-risk group: {high_risk['Diagnosis'].mode()[0]}")
    
    # Success prediction factors
    success_correlation = df[['Age', 'InitialSeverity', 'Comorbidities', 'TreatmentSuccess']].corr()['TreatmentSuccess']
    insights.append("\nSuccess Prediction Factors:")
    for factor, corr in success_correlation.items():
        if factor != 'TreatmentSuccess':
            insights.append(f"- {factor}: {'Positive' if corr > 0 else 'Negative'} correlation ({corr:.2f})")
    
    return "\n".join(insights)

def generate_report():
    """Generate the complete analysis report"""
    # Load data
    df = load_data()
    
    # Generate report sections
    sections = {
        'Demographics': generate_demographic_insights(df),
        'Diagnosis Patterns': generate_diagnosis_insights(df),
        'Treatment Effectiveness': generate_treatment_insights(df),
        'Predictive Insights': generate_predictive_insights(df)
    }
    
    # Create report
    report = []
    report.append("# HealthConnect Data Analysis Report")
    report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Executive Summary")
    report.append("Analysis of patient data reveals significant patterns in healthcare outcomes and treatment effectiveness.")
    
    for section, content in sections.items():
        report.append(f"\n## {section}")
        report.append(content)
    
    report.append("\n## Recommendations")
    report.append("Based on the analysis, we recommend:")
    report.append("1. Focus on preventive care in regions with high-risk populations")
    report.append("2. Optimize treatment plans based on success rate analysis")
    report.append("3. Implement targeted interventions for high-risk patients")
    report.append("4. Enhance follow-up care for complex cases")
    
    # Save report
    report_path = os.path.join('reports', 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Report generated and saved to: {report_path}")

if __name__ == "__main__":
    generate_report()
