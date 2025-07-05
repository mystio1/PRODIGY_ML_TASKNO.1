# House Price Prediction Project

This project implements a machine learning pipeline for predicting house prices using the Ames Housing dataset.

## 📁 Project Structure

```
HousePricePrediction/
├── main.py                 # Main ML pipeline script
├── train.csv              # Training data (replace with your data)
├── test.csv               # Test data (replace with your data)
├── sample_submission.csv  # Sample submission format (replace with your data)
├── data_description.txt   # Data description (replace with your data)
├── submission.csv         # Generated predictions (created after running)
├── target_distribution.png # Target variable analysis (created after running)
├── feature_importance.png # Feature importance plot (created after running)
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Replace Data Files
Replace the placeholder files with your actual data:
- `train.csv` - Your training data with sale prices
- `test.csv` - Your test data (without sale prices)
- `sample_submission.csv` - Your sample submission format
- `data_description.txt` - Your data description

### 2. Run the Pipeline
```bash
# From the parent directory (where ml_env is located)
ml_env\Scripts\python.exe HousePricePrediction\main.py
```

### 3. Check Results
After running, you'll get:
- `submission.csv` - Your predictions ready for submission
- `target_distribution.png` - Analysis of the target variable
- `feature_importance.png` - Most important features

## 🔧 Features

### Data Processing
- **Missing Value Handling**: Automatic imputation for numeric and categorical variables
- **Feature Encoding**: Label encoding for categorical variables
- **Data Scaling**: Standard scaling for numerical features

### Model Training
- **Multiple Algorithms**: Random Forest, Gradient Boosting, Linear Regression
- **Cross-Validation**: 5-fold cross-validation for model selection
- **Automatic Selection**: Best model selection based on RMSE

### Analysis & Visualization
- **Data Exploration**: Missing values, target distribution
- **Feature Importance**: Top features affecting house prices
- **Model Performance**: Cross-validation scores and metrics

## 📊 Expected Output

The script will output:
```
🏠 HOUSE PRICE PREDICTION PIPELINE
==================================================
Loading data...
✅ Training data loaded: (1460, 81)
✅ Test data loaded: (1459, 80)
✅ Sample submission loaded: (1459, 2)

==================================================
DATA EXPLORATION
==================================================
Training Data Info:
Shape: (1460, 81)
Columns: ['Id', 'MSSubClass', 'MSZoning', ...]

==================================================
DATA PREPROCESSING
==================================================
Handling missing values...
✅ Preprocessing completed!
   Training features: (1460, 79)
   Test features: (1459, 79)

==================================================
MODEL TRAINING
==================================================
Training Random Forest...
   CV RMSE: 25000.45 (+/- 1500.23)
Training Gradient Boosting...
   CV RMSE: 24000.12 (+/- 1200.45)
Training Linear Regression...
   CV RMSE: 28000.67 (+/- 2000.34)

✅ Best model: Gradient Boosting
   CV RMSE: 24000.12

==================================================
MAKING PREDICTIONS
==================================================
✅ Predictions completed!
   Predictions shape: (1459,)
   Submission saved as 'submission.csv'

🎉 PIPELINE COMPLETED SUCCESSFULLY!
```

## 🛠️ Customization

### Adding New Models
Edit the `train_models()` method in `main.py`:

```python
self.models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Your Model': YourModelClass()  # Add your model here
}
```

### Feature Engineering
Add your feature engineering logic in the `preprocess_data()` method:

```python
# Add new features
combined_data['TotalSF'] = combined_data['1stFlrSF'] + combined_data['2ndFlrSF']
combined_data['Age'] = combined_data['YrSold'] - combined_data['YearBuilt']
```

### Hyperparameter Tuning
Modify model parameters in the `train_models()` method:

```python
'Random Forest': RandomForestRegressor(
    n_estimators=200, 
    max_depth=10, 
    min_samples_split=5,
    random_state=42
)
```

## 📈 Model Performance

The pipeline automatically:
- Trains multiple models
- Performs cross-validation
- Selects the best performing model
- Generates feature importance analysis
- Creates visualizations for insights

## 🔍 Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure all CSV files are in the HousePricePrediction directory
   - Check file names match exactly: `train.csv`, `test.csv`, `sample_submission.csv`

2. **Memory Issues**
   - Reduce `n_estimators` in Random Forest and Gradient Boosting
   - Use fewer features or sample the data

3. **Poor Performance**
   - Add feature engineering
   - Try different algorithms
   - Tune hyperparameters
   - Check data quality

### Getting Help
- Check the console output for detailed error messages
- Verify your data format matches the expected structure
- Ensure all required libraries are installed in your ML environment

## 📚 Next Steps

1. **Improve the Model**:
   - Add more feature engineering
   - Try ensemble methods
   - Implement hyperparameter tuning

2. **Advanced Techniques**:
   - Use XGBoost or LightGBM
   - Implement stacking/blending
   - Add cross-validation with stratification

3. **Deployment**:
   - Save the trained model
   - Create a prediction API
   - Build a web interface

## 🤝 Contributing

Feel free to modify the code to suit your specific needs:
- Add new preprocessing steps
- Implement different algorithms
- Enhance visualizations
- Optimize performance

---

**Note**: This is a template project. Replace the placeholder data files with your actual dataset before running the pipeline. 