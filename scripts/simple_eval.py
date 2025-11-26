
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from price_prediction_model import PricePredictionModel

def evaluate():
    # Suppress stdout during training
    sys.stdout = open(os.devnull, 'w')
    
    try:
        model = PricePredictionModel()
        data_file = 'prediction_training_data.csv'
        
        # Load or Train
        try:
            model.load_model()
        except:
            X, y = model.prepare_training_data(data_file)
            model.train(X, y)
            
        # Evaluate
        X, y = model.prepare_training_data(data_file)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale
        X_train_scaled = model.scaler.fit_transform(X_train)
        X_test_scaled = model.scaler.transform(X_test)
        
        # Predict
        y_pred_train = model.model.predict(X_train_scaled)
        if len(y_pred_train.shape) > 1: y_pred_train = y_pred_train.flatten()
        train_acc = accuracy_score(y_train, y_pred_train)
        
        y_pred_test = model.model.predict(X_test_scaled)
        if len(y_pred_test.shape) > 1: y_pred_test = y_pred_test.flatten()
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        print(f"TRAIN_ACC:{train_acc:.4f}")
        print(f"TEST_ACC:{test_acc:.4f}")
        print(f"TRAIN_SIZE:{len(X_train)}")
        print(f"TEST_SIZE:{len(X_test)}")
        
    except Exception as e:
        sys.stdout = sys.__stdout__
        print(f"ERROR:{str(e)}")

if __name__ == "__main__":
    evaluate()
