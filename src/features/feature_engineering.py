import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import sys
import os

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def vectorize_text(train_data, test_data):
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    # memory efficient
    train_vectors = vectorizer.fit_transform(train_data['clean_comment'])
    test_vectors = vectorizer.transform(test_data['clean_comment'])
    
    # memory ineffiecient
    
    # train_vectors = train_vectors.todenses()
    # test_vectors = test_vectors.todenses()
    
    y_train = train_data['category'].values
    y_test = test_data['category'].values
    return train_vectors, test_vectors, y_train, y_test, vectorizer



def save_data(train_vectors, test_vectors, y_train, y_test, vectorizer, vectorizer_path):  #train_vectors, test_vectors
    
    os.makedirs(vectorizer_path , exist_ok=True)
    
    with open(os.path.join(vectorizer_path, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
        
    #np.save(os.path.join(vectorizer_path, 'train_vectors.npy'), train_vectors)
    #np.save(os.path.join(vectorizer_path, 'test_vectors.npy'), test_vectors)
    
    # Save sparse matrices (DO NOT convert to dense - waste of memory)
    save_npz(os.path.join(vectorizer_path, 'train_vectors.npz'), train_vectors)
    save_npz(os.path.join(vectorizer_path, 'test_vectors.npz'), test_vectors)
    
    # Save labels as numpy arrays 
    np.save(os.path.join(vectorizer_path, 'y_train.npy'), y_train)
    np.save(os.path.join(vectorizer_path, 'y_test.npy'), y_test)
    
    
    
def main():
    vectorizer_path = os.path.join("data", "interim")
    train_file = os.path.join("data", "processed", "train_data.csv")
    test_file = os.path.join("data", "processed", "test_data.csv")
    train_data, test_data = load_data(train_file, test_file)
    train_vectors, test_vectors, y_train, y_test, vectorizer = vectorize_text(train_data, test_data)
    save_data(train_vectors, test_vectors, y_train, y_test, vectorizer, vectorizer_path)
    
if __name__ == '__main__':
    main()