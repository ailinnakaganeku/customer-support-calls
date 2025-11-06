import pandas as pd
import nltk
import speech_recognition as sr
from pydub import AudioSegment
import spacy
import os
import sys
import warnings
from nltk.sentiment.vader import SentimentIntensityAnalyzer

DATA_DIR = "data"
AUDIO_FILE = os.path.join(DATA_DIR, "sample_customer_call.wav")
CSV_FILE = os.path.join(DATA_DIR, "customer_call_transcriptions.csv")

def setup_dependencies():
    """
    Downloads NLTK data and loads the spaCy model.
    """
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        print("Downloading NLTK vader_lexicon...")
        nltk.download('vader_lexicon')
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except IOError:
        print("spaCy model 'en_core_web_sm' not found.")
        print("Please run: python3 -m spacy download en_core_web_sm")
        sys.exit(1)
    return nlp

def task1_transcribe_audio(recognizer, audio_file_path):
    """
    Task 1: Transcribe audio file and print audio stats.
    """
    print(f"\n--- Task 1: Speech to Text ---")
    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at {audio_file_path}")
        return

    transcribe_audio_file = sr.AudioFile(audio_file_path)
    with transcribe_audio_file as source:
        transcribe_audio = recognizer.record(source)

    try:
        transcribed_text = recognizer.recognize_google(transcribe_audio)
        print(f"Transcribed text: {transcribed_text}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition; {e}")

    audio_segment = AudioSegment.from_file(audio_file_path)
    number_channels = audio_segment.channels
    frame_rate = audio_segment.frame_rate
    print(f"Number of channels: {number_channels}")
    print(f"Frame rate: {frame_rate}")

def task2_sentiment_analysis(df, sid):
    """
    Task 2: Perform sentiment analysis and find true positives.
    """
    print(f"\n--- Task 2: Sentiment Analysis ---")

    def find_sentiment(text):
        scores = sid.polarity_scores(str(text))
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment_predicted'] = df["text"].apply(find_sentiment)
    
    true_positive = len(df.loc[(df['sentiment_predicted'] == df['sentiment_label']) &
                               (df['sentiment_label'] == 'positive')])
    
    print(f"True positives: {true_positive}")
    return df

def task3_named_entity_recognition(df, nlp):
    """
    Task 3: Find the most frequent named entity.
    """
    print(f"\n--- Task 3: Named Entity Recognition (NER) ---")

    def extract_entities(text):
        doc = nlp(str(text))
        return [ent.text for ent in doc.ents]

    df['named_entities'] = df['text'].apply(extract_entities)
    
    all_entities = [ent for entities in df['named_entities'] for ent in entities]
    
    if not all_entities:
        print("No named entities found.")
        return df

    entities_df = pd.DataFrame(all_entities, columns=['entity'])
    entities_counts = entities_df['entity'].value_counts().reset_index()
    entities_counts.columns = ['entity', 'count']
    
    most_freq_ent = entities_counts["entity"].iloc[0]
    print(f"Most frequent entity: {most_freq_ent}")
    return df

def task4_find_similarity(df, nlp):
    """
    Task 4: Find the text most similar to a specific query.
    """
    print(f"\n--- Task 4: Text Similarity ---")
    
    warnings.filterwarnings("ignore", message=r".*\[W007\].*")

    input_query = "wrong package delivery"
    processed_query = nlp(input_query)
    
    df['processed_text'] = df['text'].apply(lambda text: nlp(str(text)))
    
    df['similarity'] = df['processed_text'].apply(lambda text: processed_query.similarity(text))
    df = df.sort_values(by='similarity', ascending=False)
    
    most_similar_text = df["text"].iloc[0]
    print(f"Most similar text to '{input_query}': {most_similar_text}")
    return df

def main():
    """
    Main function to orchestrate the analysis pipeline.
    """
    print("Starting Customer Call Analysis...")
    
    # --- Setup ---
    nlp = setup_dependencies()
    recognizer = sr.Recognizer()
    sid = SentimentIntensityAnalyzer()

    # --- Run Tasks ---
    task1_transcribe_audio(recognizer, AUDIO_FILE)
    
    if not os.path.exists(CSV_FILE):
        print(f"\nError: CSV file not found at {CSV_FILE}")
        print("Skipping Tasks 2, 3, and 4.")
        return
        
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading {CSV_FILE}: {e}")
        return

    df = task2_sentiment_analysis(df, sid)
    df = task3_named_entity_recognition(df, nlp)
    df = task4_find_similarity(df, nlp)
    
    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()