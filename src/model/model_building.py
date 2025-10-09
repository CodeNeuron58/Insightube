import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.sparse import load_npz
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(vectorized_path):
    X_train = load_npz(os.path.join(vectorized_path, "train_vectors.npz"))
    y_train = np.load(os.path.join(vectorized_path, "y_train.npy"))
    return X_train, y_train


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_valid, y_train, y_valid


def load_vectorizer(vectorized_path):
    with open(os.path.join(vectorized_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


def model_building(X_train, y_train, X_val, y_val):
    learning_rate = 0.09
    max_depth = 20
    n_estimators = 367

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric="multi_logloss",
        is_unbalance=True,
        class_weight="balanced",
        reg_alpha=0.01,
        reg_lambda=0.01,
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        force_col_wise=True
    )

    model.fit(X_train, y_train)
    return model
    
    
def save_model(model, model_path = 'models/model.pkl'):
    

    # Save the trained model
    os.makedirs('models', exist_ok=True)
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)


def main():
    vectorized_path = os.path.join("data", "interim")
    X_train, y_train = load_data(vectorized_path)
    X_train, X_val, y_train, y_val = split_data(X_train, y_train)

    model = model_building(X_train, y_train, X_val, y_val)
    save_model(model)

if __name__ == "__main__":
    main()
