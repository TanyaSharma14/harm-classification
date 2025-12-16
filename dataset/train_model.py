import pandas as pd
import nltk
import joblib
import random

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

nltk.download('stopwords')
from nltk.corpus import stopwords

# ===============================
# LOAD DATASET (HARMFUL VIDEOS)
# ===============================
df = pd.read_csv("Harmful _full_agreement.csv")

# Use transcript as harmful text
harmful_df = df[['transcript']].dropna()
harmful_df['label'] = 1

# ===============================
# CREATE NON-HARMFUL SAMPLES
# ===============================
# Use titles & descriptions as safe content
safe_texts = []

for col in ['title', 'description']:
    if col in df.columns:
        safe_texts.extend(df[col].dropna().tolist())

# Randomly sample safe texts
safe_texts = random.sample(safe_texts, min(len(safe_texts), len(harmful_df)))

safe_df = pd.DataFrame({
    'text': safe_texts,
    'label': 0
})

# Rename harmful column
harmful_df = harmful_df.rename(columns={'transcript': 'text'})

# ===============================
# COMBINE DATA
# ===============================
final_df = pd.concat([harmful_df, safe_df], ignore_index=True)

# ===============================
# TEXT CLEANING
# ===============================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    words = text.split()
    return " ".join(w for w in words if w.isalpha() and w not in stop_words)

final_df['text'] = final_df['text'].apply(clean_text)

X = final_df['text']
y = final_df['label']

print("Class distribution:")
print(y.value_counts())

# ===============================
# TF-IDF
# ===============================
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3
)

X_vec = vectorizer.fit_transform(X)

# ===============================
# TRAIN / TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# MODEL TRAINING
# ===============================
model = LinearSVC()
model.fit(X_train, y_train)

# ===============================
# SAVE MODEL
# ===============================
joblib.dump(model, "harm_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model trained successfully")
print("📦 Files saved")
