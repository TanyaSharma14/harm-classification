YouTube Harmful Content Detection using BERT
Overview

This project detects harmful or abusive content in YouTube videos by analyzing their transcripts, or when transcripts are unavailable, the video title and description.
It combines a fine-tuned BERT-based classifier with keyword-based severity adjustment to provide a final harm classification.

The system outputs:

Harm category: NOT HARMFUL, MODERATE, or HARMFUL

A continuous severity score

Detected harmful words (if any)

Motivation

Online video platforms face challenges in moderating harmful speech at scale. Manual moderation does not scale, and rule-based systems fail on contextual language.

This project explores:

Transformer-based text classification

Real-world noisy data (YouTube transcripts)

Trade-offs between ML predictions and heuristic signals

Tech Stack

Python

PyTorch

HuggingFace Transformers

BERT (bert-base-uncased)

scikit-learn

YouTube Transcript API

BeautifulSoup

Google Colab

 Dataset

Harmful content: Harmful.csv containing harmful transcripts

Safe content: Synthetic neutral sentences created to balance classes

Labels:

1 → Harmful

0 → Safe

The dataset is stratified during train-test split to maintain class balance.

 Model Training

Tokenizer: bert-base-uncased

Max sequence length: 256

Epochs: 3

Batch size: 8

Loss: Cross-Entropy (default in BertForSequenceClassification)

The model is fine-tuned end-to-end using HuggingFace’s Trainer API
