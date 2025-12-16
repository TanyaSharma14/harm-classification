import joblib
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

model = joblib.load("harm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# LOAD CATEGORY WORDS
categories = {}
current = None

with open("harmful_words.txt") as f:
    for line in f:
        line = line.strip().lower()
        if line.startswith("["):
            current = line[1:-1]
            categories[current] = []
        elif line and current:
            categories[current].append(line)

def extract_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    if "/live/" in url:
        return url.split("/live/")[1].split("?")[0]
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    raise ValueError("Invalid YouTube URL")

def get_text(vid):
    try:
        t = YouTubeTranscriptApi.get_transcript(vid)
        return " ".join(x['text'] for x in t), "Transcript"
    except:
        html = requests.get(f"https://www.youtube.com/watch?v={vid}").text
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.text.replace("- YouTube", "")
        desc = soup.find("meta", {"name": "description"})
        return title + " " + (desc["content"] if desc else ""), "Title+Description"

def analyze(text):
    text_l = text.lower()
    found = {}
    score = 0

    for cat, words in categories.items():
        hits = [w for w in words if w in text_l]
        if hits:
            found[cat] = hits
            score += 0.25 * len(hits)

    return found, min(score, 1.0)

url = input("Enter YouTube video URL: ")
vid = extract_video_id(url)

text, source = get_text(vid)

X = vectorizer.transform([text])
prob = model.predict_proba(X)[0][1]

found, keyword_score = analyze(text)

final_score = min((prob * 0.6) + (keyword_score * 0.4), 1.0)

if final_score < 0.2:
    level = "NOT HARMFUL"
elif final_score < 0.5:
    level = "MILD"
elif final_score < 0.75:
    level = "HARMFUL"
else:
    level = "VERY HARMFUL"
print("\n========== RESULT ==========")
print("Text source:", source)
print("Severity Level:", level)
print("Risk Score:", round(final_score, 2))
print("Categories Detected:", list(found.keys()))
print("Trigger Words:", found)
