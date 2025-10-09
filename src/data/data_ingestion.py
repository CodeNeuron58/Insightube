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


def split_data(df, test_size=0.2, random_state=42):
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_data, test_data


def save_data(train_data, test_data, data_path):
    os.makedirs(data_path , exist_ok=True)
    
    train_data.to_csv(os.path.join(data_path, "train_data.csv"), index=False)
    test_data.to_csv(os.path.join(data_path, "test_data.csv"), index=False)
    
    
def main():
    data_path = os.path.join("data", "raw")
    df = load_data("https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")
    df = preprocess_data(df)
    train_data, test_data = split_data(df)
    save_data(train_data, test_data, data_path)
    
    
if __name__ == "__main__":
    main()