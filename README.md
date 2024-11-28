# HealthConnect Data Analysis Project

## Project Overview
This project analyzes patient data for HealthConnect, a healthcare solutions provider, to discover trends and insights in patient outcomes, demographics, and treatment effectiveness.

## Project Structure
```
├── data/               # Data directory for raw and processed data
├── notebooks/          # Jupyter notebooks for analysis
├── scripts/           # Python scripts for data processing
├── visualizations/    # Exported visualizations and Tableau workbooks
├── reports/          # Final reports and presentations
├── Dockerfile        # Docker configuration
└── docker-compose.yml # Docker Compose configuration
└── environment.yml    # Conda environment configuration
```

## Setup Instructions (Using Docker)

1. Build and start the container:
```bash
docker-compose up --build
```

2. Access Jupyter Notebook:
   - Open your browser and go to: http://localhost:8888
   - Token: healthconnect

3. Start analyzing the data:
   - Open `notebooks/healthcare_analysis.ipynb`
   - All required packages are pre-installed
   - Synthetic data will be automatically generated

## Project Components
1. Data Cleaning and Preprocessing
2. Exploratory Data Analysis
3. Statistical Analysis
4. Tableau Visualizations
5. Predictive Analytics
6. Final Report and Presentation

## Tools Used
- Python (Pandas, NumPy, Scikit-learn)
- Jupyter Notebook
- Tableau
- Matplotlib/Seaborn
- Docker
