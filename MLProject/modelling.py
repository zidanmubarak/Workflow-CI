"""
MLflow Project - Climate Change Agriculture Model Training
Dataset: Climate Change Impact on Agriculture 2024
Author: Zidan Mubarak
Description: Automated model training with MLflow Project for CI/CD
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import argparse
from datetime import datetime

def load_and_prepare_data(data_path):
    """Load and prepare the preprocessed dataset"""
    print(f"\n📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    
    # Define features (exclude target and non-numeric columns)
    exclude_cols = ['Crop_Yield_MT_per_HA', 'Country', 'Region', 'Crop_Type', 'Unnamed: 0']
    feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]
    
    # Create binary classification target
    median_yield = df['Crop_Yield_MT_per_HA'].median()
    df['Yield_Category'] = (df['Crop_Yield_MT_per_HA'] > median_yield).astype(int)
    
    X = df[feature_cols]
    y = df['Yield_Category']
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols

def create_confusion_matrix_plot(y_true, y_pred, save_path='confusion_matrix.png'):
    """Create and save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - Crop Yield Prediction', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {save_path}")
    return save_path

def create_feature_importance_plot(model, feature_names, save_path='feature_importance.png', top_n=15):
    """Create and save feature importance plot"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Feature importance plot saved to {save_path}")
    return save_path

def save_classification_report(y_true, y_pred, save_path='classification_report.txt'):
    """Save classification report to text file"""
    report = classification_report(y_true, y_pred)
    with open(save_path, 'w') as f:
        f.write("Classification Report - Crop Yield Prediction\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Classification report saved to {save_path}")
    return save_path

def train_model(data_path, n_estimators=100, max_depth=10, test_size=0.2, random_state=42):
    """
    Train Random Forest model with MLflow tracking
    
    Parameters:
    -----------
    data_path : str
        Path to preprocessed CSV file
    n_estimators : int
        Number of trees in random forest
    max_depth : int
        Maximum depth of trees
    test_size : float
        Test set ratio
    random_state : int
        Random seed
    """
    
    print("="*70)
    print("MLFLOW PROJECT - CLIMATE CHANGE AGRICULTURE MODEL")
    print("="*70)
    
    # Load data
    X, y, feature_names = load_and_prepare_data(data_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n📊 Training set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"RF_CI_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("dataset_path", data_path)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", len(feature_names))
        
        # Train model
        print(f"\n🔧 Training Random Forest...")
        print(f"   n_estimators: {n_estimators}")
        print(f"   max_depth: {max_depth}")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Print metrics
        print(f"\n📈 Model Performance:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        
        # Create and log artifacts
        print(f"\n📊 Creating artifacts...")
        
        # Confusion matrix
        cm_path = create_confusion_matrix_plot(y_test, y_pred)
        mlflow.log_artifact(cm_path)
        
        # Feature importance
        fi_path = create_feature_importance_plot(model, feature_names)
        mlflow.log_artifact(fi_path)
        
        # Classification report
        cr_path = save_classification_report(y_test, y_pred)
        mlflow.log_artifact(cr_path)
        
        # Save and log model
        print(f"\n💾 Saving model...")
        
        # Create model directory
        os.makedirs("model", exist_ok=True)
        model_path = "model/random_forest_model.pkl"
        
        # Save model as pickle
        joblib.dump(model, model_path)
        print(f"   Model saved to: {model_path}")
        
        # Log model as artifact
        mlflow.log_artifact(model_path)
        
        # Try to log with mlflow.sklearn
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn_model"
            )
            print(f"   Model logged with mlflow.sklearn")
        except Exception as e:
            print(f"   Note: mlflow.sklearn logging skipped")
        
        # Log tags
        mlflow.set_tag("model_type", "Random Forest")
        mlflow.set_tag("dataset", "Climate Change Agriculture")
        mlflow.set_tag("author", "Zidan Mubarak")
        mlflow.set_tag("ci_cd", "GitHub Actions")
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Training completed!")
        print(f"   MLflow Run ID: {run_id}")
        
        return model, run_id

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Climate Change Agriculture Model')
    parser.add_argument('--data-path', type=str, default='climate_change_preprocessing.csv',
                        help='Path to preprocessed dataset')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in random forest')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Maximum depth of trees')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set ratio')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Train model
    model, run_id = train_model(
        data_path=args.data_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print("\n" + "="*70)
    print("✅ MLflow Project execution completed!")
    print("="*70)
