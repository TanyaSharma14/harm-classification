# YouTube Harmful Content Detection (BERT + Keyword Severity)

This project fine-tunes a **BERT** binary classifier to detect whether text is **harmful** or **not harmful**, then applies it to a YouTube video by extracting its **transcript** (or fallback to **title + description**) and computing a final **severity score** using:
- BERT probability of “harmful”
- count of matched harmful keywords from a wordlist (`en.txt`)

---

## What this does

### 1) Model training (binary classification)
- Reads `Harmful.csv` and uses the `transcript` column as harmful samples (`label = 1`)
- Creates synthetic safe samples (`label = 0`) using repeated neutral sentences
- Splits into train/test using stratified split
- Tokenizes with `bert-base-uncased`
- Fine-tunes `BertForSequenceClassification` for 2 classes
- Saves model + tokenizer to `bert_model/`

### 2) YouTube video scoring
Given a YouTube URL:
- Extracts the video ID
- Tries to fetch transcript using `youtube_transcript_api`
- If transcript fails, uses **title + meta description** by scraping the page
- Runs BERT on the gathered text
- Loads harmful keywords from `en.txt` and counts matches
- Computes severity:

- Outputs:
- text source (Transcript vs Title+Description)
- final label (NOT HARMFUL / MODERATE / HARMFUL)
- severity score
- detected harmful words

---

## Project structure (recommended)

