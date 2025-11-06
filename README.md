# Customer Support Calls
The main script (`main.py`) performs 4 key tasks:
1.  **Speech-to-Text**: Transcribes a sample `.wav` audio file into text.
2.  **Sentiment Analysis**: Uses NLTK (VADER) to classify transcriptions and find the number of "true positives."
3.  **Named Entity Recognition (NER)**: Uses spaCy to extract and find the most common entities (like people, places, or dates).
4.  **Similarity Search**: Finds the most semantically similar transcription to a specific query (`"wrong package delivery"`).