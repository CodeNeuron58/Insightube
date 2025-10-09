from flask import Flask, request, jsonify, render_template
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os


app = Flask(__name__)

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment
    
def load_model(model_path = 'models/model.pkl'):
    with open((model_path), 'rb') as f:
        model = pickle.load(f)
    return model

def load_vectorizer(vectorized_path = 'data/interim'):
    with open(os.path.join(vectorized_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

vectorizer = load_vectorizer()
model = load_model()

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = preprocess_comment(text)
    text = vectorizer.transform([text])
    text = model.predict(text)
    return  render_template('home.html', prediction=text[0]) # Return the predicted text

if __name__ == '__main__':
    app.run(debug=True)



