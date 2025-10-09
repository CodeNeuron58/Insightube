import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
import sys
import os


def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df["clean_comment"].str.strip() != ""]
    return df


def save_data(train_data, test_data, data_path):
    os.makedirs(data_path , exist_ok=True)
    
    train_data.to_csv(os.path.join(data_path, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(data_path, "test_data.csv"), index=False)
    
    
def main():
    data_path = os.path.join("data", "processed")
    train_data_path = os.path.join("data", "processed", "train_data.csv")
    test_data_path = os.path.join("data", "processed", "test_data.csv")
    
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)
    
    save_data(train_data, test_data, data_path)
    
if __name__ == '__main__':
    main()
    