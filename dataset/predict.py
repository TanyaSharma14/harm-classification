import joblib
import re
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

# ===============================
# LOAD MODEL & VECTORIZER
# ===============================
model = joblib.load("harm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ===============================
# LOAD HARMFUL WORDS
# ===============================
with open("harmful_words.txt", "r", encoding="utf-8") as f:
    harmful_words = set(f.read().lower().split())

# ===============================
# FUNCTIONS
# ===============================
def extract_video_id(url):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    if not match:
        raise Exception("❌ Invalid YouTube URL")
    return match.group(1)

def get_text_from_youtube(video_id):
    """
    Try transcript first.
    If not available, fallback to title + description.
    """
    # 1️⃣ Try transcript
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([x['text'] for x in transcript]), "Transcript"
    except Exception:
        pass

    # 2️⃣ Fallback: title + description (safe)
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://www.youtube.com/watch?v={video_id}"
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")

        title = soup.find("title").text.replace("- YouTube", "")
        description = ""
        meta_desc = soup.find("meta", {"name": "description"})
        if meta_desc:
            description = meta_desc.get("content", "")

        return title + " " + description, "Title + Description"
    except Exception:
        raise Exception("❌ Could not extract any text from video")

def find_harmful_words(text):
    words = set(text.lower().split())
    return sorted(list(words & harmful_words))

# ===============================
# MAIN
# ===============================
url = input("Enter YouTube video URL: ")
video_id = extract_video_id(url)

text, source = get_text_from_youtube(video_id)

X = vectorizer.transform([text])
prediction = model.predict(X)[0]

harm_words = find_harmful_words(text)

print("\n========== RESULT ==========")
print("📄 Text source used:", source)

if prediction == 1:
    print("❌ Video is HARMFUL")
else:
    print("✅ Video is NOT HARMFUL")

print("⚠ Harmful words found:", harm_words)
