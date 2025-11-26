
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from price_prediction_model import PricePredictionModel

def train_and_save():
    print("="*80)
    print("TRAINING AND SAVING MODEL")
    print("="*80)
    
    data_file = 'prediction_training_data.csv'
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found.")
        return

    model = PricePredictionModel()
    X, y = model.prepare_training_data(data_file)
    
    print(f"Training on {len(X)} samples...")
    model.train(X, y)
    
    model.save_model()
    print("Model trained and saved.")

if __name__ == "__main__":
    train_and_save()
