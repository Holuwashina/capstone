import nbconvert
import os

def convert_notebook_to_html():
    """Convert the analysis notebook to HTML with a table of contents"""
    os.system('jupyter nbconvert --to html --template classic \
              --ExecutePreprocessor.timeout=600 \
              --execute notebooks/complete_healthcare_analysis.ipynb \
              --output ../reports/HealthConnect_Analysis.html')

if __name__ == "__main__":
    convert_notebook_to_html()
