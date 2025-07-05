#!/usr/bin/env python3
"""
Test script to verify ML environment setup
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def test_imports():
    """Test if all required libraries can be imported"""
    print("Testing library imports...")
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__}")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    return True

def test_basic_ml_pipeline():
    """Test a basic ML pipeline"""
    print("\nTesting basic ML pipeline...")
    
    try:
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                 n_redundant=2, random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ ML pipeline completed successfully!")
        print(f"   Model accuracy: {accuracy:.4f}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ ML pipeline failed: {e}")
        return False

def test_visualization():
    """Test basic visualization capabilities"""
    print("\nTesting visualization capabilities...")
    
    try:
        # Create sample data
        data = np.random.randn(100, 2)
        df = pd.DataFrame(data, columns=['x', 'y'])
        
        # Create a simple plot
        plt.figure(figsize=(8, 6))
        plt.scatter(df['x'], df['y'], alpha=0.6)
        plt.title('Sample Scatter Plot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig('test_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization test completed!")
        print("   Plot saved as 'test_plot.png'")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("ML Environment Test Suite")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Import tests failed. Please check your installation.")
        sys.exit(1)
    
    # Test ML pipeline
    ml_ok = test_basic_ml_pipeline()
    
    # Test visualization
    viz_ok = test_visualization()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Library Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"ML Pipeline: {'✅ PASS' if ml_ok else '❌ FAIL'}")
    print(f"Visualization: {'✅ PASS' if viz_ok else '❌ FAIL'}")
    
    if all([imports_ok, ml_ok, viz_ok]):
        print("\n🎉 All tests passed! Your ML environment is ready to use.")
        print("\nNext steps:")
        print("1. Start Jupyter: jupyter notebook")
        print("2. Open notebooks/01_getting_started.ipynb")
        print("3. Begin your ML project!")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 