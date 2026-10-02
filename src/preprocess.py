import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]
        
    os.makedirs('data/processed', exist_ok=True)
    
    # Load raw data
    train_data = np.load('data/raw/train.npz')
    test_data = np.load('data/raw/test.npz')
    
    x_train_raw, y_train_raw = train_data['x'], train_data['y']
    x_test, y_test = test_data['x'], test_data['y']
    
    # Normalize pixel values to [0, 1]
    x_train_raw = x_train_raw.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Split train/val
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_raw, y_train_raw, 
        test_size=params["test_size"], 
        random_state=params["seed"]
    )
    
    # Save processed arrays
    np.savez_compressed('data/processed/train.npz', x=x_train, y=y_train)
    np.savez_compressed('data/processed/val.npz', x=x_val, y=y_val)
    np.savez_compressed('data/processed/test.npz', x=x_test, y=y_test)
    print("Processed data saved to data/processed/")

if __name__ == '__main__':
    preprocess()
