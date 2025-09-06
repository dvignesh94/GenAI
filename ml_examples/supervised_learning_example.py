"""
Supervised Learning Example: Classification and Regression

This example demonstrates both classification and regression using scikit-learn.
We'll use the famous Iris dataset for classification and Boston housing dataset for regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import seaborn as sns

def classification_example():
    """Example of supervised learning for classification"""
    print("=" * 50)
    print("SUPERVISED LEARNING: CLASSIFICATION")
    print("=" * 50)
    
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Feature importance
    feature_importance = clf.feature_importances_
    print("\nFeature Importance:")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance')
    plt.xticks(rotation=45)
    
    # Visualize predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([0, 2], [0, 2], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predictions vs Actual')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/supervised_classification_results.png')
    plt.show()
    
    return clf, accuracy

def regression_example():
    """Example of supervised learning for regression"""
    print("\n" + "=" * 50)
    print("SUPERVISED LEARNING: REGRESSION")
    print("=" * 50)
    
    # Load the Boston housing dataset
    boston = load_boston()
    X, y = boston.data, boston.target
    feature_names = boston.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target: Median value of owner-occupied homes in $1000's")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train a Random Forest Regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)
    
    # Make predictions
    y_pred = reg.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = reg.feature_importances_
    print("\nTop 5 Most Important Features:")
    importance_pairs = list(zip(feature_names, feature_importance))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)
    for name, importance in importance_pairs[:5]:
        print(f"{name}: {importance:.4f}")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Feature importance
    plt.subplot(1, 3, 1)
    top_features = importance_pairs[:8]
    names, importances = zip(*top_features)
    plt.barh(names, importances)
    plt.title('Top 8 Feature Importance')
    plt.xlabel('Importance')
    
    # Predictions vs actual
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Price ($1000s)')
    plt.ylabel('Predicted Price ($1000s)')
    plt.title('Predictions vs Actual')
    plt.legend()
    
    # Residuals
    plt.subplot(1, 3, 3)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price ($1000s)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig('/Users/vignesh/Documents/GitHub/Generative AI/ml_examples/supervised_regression_results.png')
    plt.show()
    
    return reg, r2

def main():
    """Main function to run both examples"""
    print("SUPERVISED LEARNING EXAMPLES")
    print("This example demonstrates both classification and regression")
    print("using the Iris and Boston housing datasets.\n")
    
    # Run classification example
    clf, clf_accuracy = classification_example()
    
    # Run regression example
    reg, reg_r2 = regression_example()
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Classification Accuracy: {clf_accuracy:.4f}")
    print(f"Regression R² Score: {reg_r2:.4f}")
    print("\nSupervised learning uses labeled data to train models that can")
    print("make predictions on new, unseen data. The model learns the")
    print("relationship between input features and target outputs.")

if __name__ == "__main__":
    main()
