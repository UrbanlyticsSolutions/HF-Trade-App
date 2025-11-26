
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from price_prediction_model import PricePredictionModel

def evaluate_model():
    print("="*80)
    print("EVALUATING PRICE PREDICTION MODEL")
    print("="*80)
    
    data_file = 'prediction_training_data.csv'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    # Initialize model
    model = PricePredictionModel()
    
    # Try to load existing model first
    try:
        model.load_model()
        print("Loaded existing model.")
    except:
        print("Could not load model. Training new model for evaluation...")
        # If no model exists, we train one to see potential accuracy
        X, y = model.prepare_training_data(data_file)
        metrics = model.train(X, y)
        print(f"New Model Test Accuracy: {metrics['test_accuracy']:.2%}")
        return

    # If model loaded, evaluate it on the data
    print("Evaluating loaded model on data...")
    X, y = model.prepare_training_data(data_file)
    
    # We use the same split as training to be fair (or just evaluate on all if we want to see fit)
    # But typically we want to see test accuracy. 
    # Since we don't know the exact split used for the saved model, 
    # we will split again and see how it performs on the test set.
    # Note: This might overlap with training data if the seed was different, 
    # but it gives a good estimate of performance.
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData Split:")
    print(f"  Train Set: {len(X_train)} samples (80%)")
    print(f"  Test Set:  {len(X_test)} samples (20%)")
    
    # Scale
    X_train_scaled = model.scaler.fit_transform(X_train)
    X_test_scaled = model.scaler.transform(X_test)
    
    # Predict Train
    y_pred_train = model.model.predict(X_train_scaled)
    if len(y_pred_train.shape) > 1:
        y_pred_train = y_pred_train.flatten()
    train_acc = accuracy_score(y_train, y_pred_train)
    
    # Predict Test
    y_pred_test = model.model.predict(X_test_scaled)
    if len(y_pred_test.shape) > 1:
        y_pred_test = y_pred_test.flatten()
    test_acc = accuracy_score(y_test, y_pred_test)
    
    output = []
    output.append("="*80)
    output.append("EVALUATION RESULTS")
    output.append("="*80)
    output.append(f"Data Split:")
    output.append(f"  Train Set: {len(X_train)} samples (80%)")
    output.append(f"  Test Set:  {len(X_test)} samples (20%)")
    output.append(f"")
    output.append(f"Results:")
    output.append(f"  Train Accuracy: {train_acc:.2%}")
    output.append(f"  Test Accuracy:  {test_acc:.2%}")
    output.append(f"")
    output.append("Test Classification Report:")
    output.append(classification_report(y_test, y_pred_test, target_names=['DOWN', 'NEUTRAL', 'UP']))
    
    with open('evaluation_results.txt', 'w') as f:
        f.write('\n'.join(output))
        
    print('\n'.join(output))

if __name__ == "__main__":
    evaluate_model()
