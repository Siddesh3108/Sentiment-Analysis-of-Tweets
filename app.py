import numpy as np
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from flask import Flask, render_template, request

import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__)


nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


df = pd.read_csv("twitter_training.csv")
df.dropna(inplace=True)


def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(filtered_tokens)


df['Preprocessed Text'] = df['Comment'].apply(preprocess)


le_model = LabelEncoder()
df['Sentiment'] = le_model.fit_transform(df['Sentiment'])


x_train, x_test, y_train, y_test = train_test_split(
    df['Preprocessed Text'], df['Sentiment'], test_size=0.2, random_state=42, stratify=df['Sentiment']
)


nb_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('naive_bayes', MultinomialNB())
])
nb_pipeline.fit(x_train, y_train)
joblib.dump(nb_pipeline, 'nb_pipeline.pkl') 

rf_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('random_forest', RandomForestClassifier())
])
rf_pipeline.fit(x_train, y_train)
joblib.dump(rf_pipeline, 'rf_pipeline.pkl') 


nb_pipeline = joblib.load('nb_pipeline.pkl')
rf_pipeline = joblib.load('rf_pipeline.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    true_label = None

    if request.method == 'POST':
        comment = request.form['comment']
        processed_comment = [preprocess(comment)]
        selected_model = request.form['model']

        if selected_model == 'naive_bayes':
            predicted_label = nb_pipeline.predict(processed_comment)
        else:
            predicted_label = rf_pipeline.predict(processed_comment)

        classes = ['Irrelevant', 'Neutral', 'Negative', 'Positive']
        prediction = classes[predicted_label[0]]

        
        test_df = pd.read_csv('twitter_validation.csv')
        true_label = test_df['Sentiment'][11]  

    return render_template('index.html', prediction=prediction, true_label=true_label)

if __name__ == '__main__':
    app.run(debug=True)
