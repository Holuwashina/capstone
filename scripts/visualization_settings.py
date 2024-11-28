# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Basic seaborn setup without any complex styles
sns.set_theme(style='white')
sns.set_palette('husl')

# Set figure size
plt.rcParams.update({
    'figure.figsize': [12, 8],
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12
})
