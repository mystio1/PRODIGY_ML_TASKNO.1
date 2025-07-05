# ML Model Development Environment

This project provides a complete Python environment for machine learning model development.

## 🚀 Quick Start

### 1. Activate the Virtual Environment
```bash
# On Windows (PowerShell)
ml_env\Scripts\activate

# If you get execution policy errors, use:
ml_env\Scripts\python.exe your_script.py
```

### 2. Verify Installation
```bash
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, jupyter; print('✅ All libraries installed successfully!')"
```

### 3. Start Jupyter Notebook
```bash
jupyter notebook
# or
jupyter lab
```

## 📦 Installed Libraries

### Core ML Libraries
- **NumPy** (2.3.1) - Numerical computing
- **Pandas** (2.3.0) - Data manipulation and analysis
- **Scikit-learn** (1.7.0) - Machine learning algorithms
- **Matplotlib** (3.10.3) - Plotting and visualization
- **Seaborn** (0.13.2) - Statistical data visualization

### Development Tools
- **Jupyter** (1.1.1) - Interactive notebooks
- **IPython** (9.3.0) - Enhanced Python shell

## 📁 Project Structure
```
├── ml_env/                 # Virtual environment
├── data/                   # Data files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🛠️ Usage Examples

### Basic Data Analysis
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/your_data.csv')

# Basic analysis
print(df.head())
print(df.describe())

# Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True)
plt.show()
```

### Machine Learning Pipeline
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Prepare data
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
```

## 🔧 Troubleshooting

### PowerShell Execution Policy
If you encounter execution policy errors, you can:
1. Use the full path to python: `ml_env\Scripts\python.exe`
2. Or change execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Package Installation Issues
If you need to install additional packages:
```bash
ml_env\Scripts\python.exe -m pip install package_name
```

## 📚 Next Steps

1. Create your first notebook in the `notebooks/` directory
2. Add your data files to the `data/` directory
3. Start building your ML models!

## 🤝 Contributing

Feel free to add more libraries to `requirements.txt` as needed for your specific ML project. 