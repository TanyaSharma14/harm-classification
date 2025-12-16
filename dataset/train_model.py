import pandas as pd
import nltk
import joblib
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
nltk.download("stopwords")

df = pd.read_csv("Harmful _full_agreement.csv")

harmful = df[['transcript']].dropna()
harmful['label'] = 1
harmful.columns = ['text', 'label']

neutral = []
for c in ['title', 'description']:
    if c in df.columns:
        neutral.extend(df[c].dropna().tolist())

neutral = pd.DataFrame({
    'text': neutral[:len(harmful)],
    'label': 0
})

safe = pd.DataFrame({
    'text': [
        "educational content",
        "learning tutorial",
        "helpful explanation",
        "good lecture",
        "useful information"
    ] * (len(harmful)//5),
    'label': 0
})

data = pd.concat([harmful, neutral, safe], ignore_index=True)

stop_words = set(stopwords.words("english"))

def clean(t):
    return " ".join(
        w for w in str(t).lower().split()
        if w.isalpha() and w not in stop_words
    )

data['text'] = data['text'].apply(clean)

X = data['text']
y = data['label']

vectorizer = TfidfVectorizer(
    max_features=40000,
    ngram_range=(1,2)
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)
joblib.dump(model, "harm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("Powerful probabilistic model trained")
