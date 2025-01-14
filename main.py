from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict
import csv
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

app = FastAPI()

emotion_lexicon = {
    'joy': ['happy', 'excited', 'elated', 'joyful', 'cheerful'],
    'anger': ['angry', 'frustrated', 'irritated', 'enraged'],
    'sadness': ['sad', 'depressed', 'down', 'gloomy', 'lonely'],
    'fear': ['scared', 'fearful', 'anxious', 'nervous', 'terrified'],
    'love': ['love', 'adore', 'affectionate', 'fond', 'heartfelt'],
}

def preprocess_text(text: str) -> str:
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

def score_emotions(text: str) -> Dict[str, int]:
    words = text.split()
    emotion_scores = defaultdict(int)
    
    # Score based on lexicon
    for word in words:
        for emotion, emotion_words in emotion_lexicon.items():
            if word in emotion_words:
                emotion_scores[emotion] += 1
    
    return emotion_scores

# Normalize emotion scores to ensure the sum equals 100
def normalize_emotions(emotion_scores: Dict[str, int]) -> Dict[str, int]:
    total_score = sum(emotion_scores.values())
    if total_score > 0:
        normalization_factor = 100 / total_score
        for emotion in emotion_scores:
            emotion_scores[emotion] = round(emotion_scores[emotion] * normalization_factor)
    return emotion_scores

# Function to train the sentiment analysis model
def train_model(csv_file: str):
    texts = []
    labels = []
    
    # Read the CSV file
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            text, label = row
            texts.append(preprocess_text(text))
            labels.append(int(label))
    

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy * 100:.2f}%')
    
    return vectorizer, model

vectorizer, model = train_model('data.csv')

# Function to predict the sentiment of a new text
def predict_sentiment(text: str) -> Dict[str, int]:
    # Preprocess the input text
    text = preprocess_text(text)
    
    # Transform the text to bag-of-words features
    X_new = vectorizer.transform([text])
    
    # Predict the sentiment label (0: sadness, 1: joy, 2: love, 3: anger, 4: fear)
    label = model.predict(X_new)[0]
    
    emotions = {
        "sadness": 0,
        "joy": 0,
        "love": 0,
        "anger": 0,
        "fear": 0
    }


    if label == 0:
        emotions["sadness"] = 50
        emotions["fear"] = 30
        emotions["anger"] = 20
    elif label == 1:
        emotions["joy"] = 60
        emotions["joy"] = 40
    elif label == 2:
        emotions["love"] = 70
        emotions["joy"] = 30
    elif label == 3:
        emotions["anger"] = 50
        emotions["fear"] = 30
    elif label == 4:
        emotions["fear"] = 50
        emotions["fear"] = 50

    return normalize_emotions(emotions)

class JournalEntry(BaseModel):
    entry: str

@app.post("/analyze/")
async def analyze_journal(entry: JournalEntry):
    emotions = predict_sentiment(entry.entry)
    
    return emotions


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
