from scipy.sparse import load_npz
import numpy as np
import os

import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def load_data(vectorized_path):
    X_test = load_npz(os.path.join(vectorized_path, "test_vectors.npz"))
    y_test = np.load(os.path.join(vectorized_path, "y_test.npy"))
    return X_test, y_test

def load_model(model_path):
    with open((model_path), 'rb') as f:
        model = pickle.load(f)
    return model

def load_vectorizer(vectorized_path):
    with open(os.path.join(vectorized_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def main():
    model_path = os.path.join("models", "model.pkl")
    vectorized_path = os.path.join("data", "interim")
    X_test, y_test = load_data(vectorized_path)
    model = load_model(model_path)
    evaluate_model(model, X_test, y_test)
    
if __name__ == "__main__":
    main()