import speech_recognition as sr
import pyttsx3
import pyaudio
import wave
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import threading

# Parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of channels
RATE = 44100  # Sample rate (Hz)
CHUNK = 1024  # Chunk size (number of frames per buffer)
RECORD_SECONDS = 10  # Duration of each recording (seconds)
OUTPUT_FILENAME = "recorded_audio.wav"  # Output filename

# Initialize the recognizer
r = sr.Recognizer()

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained BART model and tokenizer for summarization
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Loading the dataset of keywords
datapath = r"C:\Users\Lenovo\Documents\Rohit_AI_ML\SummariseIT\dataset.csv"
df = pd.read_csv(datapath)
# Flatten the dataset to create a set of valid keywords
valid_keywords = set()
for column in df.columns:
    valid_keywords.update(df[column].dropna().str.strip().tolist())

def extract_keywords_from_tokens(text, model, tokenizer, num_keywords=5):
    """Extract keywords from the text using BERT embeddings."""
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Get the [CLS] token's embedding
    cls_embedding = last_hidden_state[:, 0, :].squeeze()
    
    # Calculate similarity between each token embedding and the [CLS] embedding
    similarities = torch.matmul(last_hidden_state.squeeze(), cls_embedding)
    
    # Get the indices of the top-n tokens with the highest similarity
    top_indices = similarities.topk(num_keywords).indices

    # Extract the corresponding tokens, excluding [CLS] and checking if they are in valid_keywords
    keywords = [tokens[i] for i in top_indices if tokens[i] != '[CLS]' and tokens[i] in valid_keywords]
    
    return keywords

def generate_summary(text, model, tokenizer):
    """Generate a summary using BART model."""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def process_transcription(text):
    """Process the transcribed text: extract keywords and generate summaries."""
    # Save the transcription to a text file
    with open("transcription.txt", "w") as f:
        f.write(text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove punctuation and make lowercase
    words = [word.lower() for word in words if word.isalnum()]

    # Remove stop words
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]

    # Count the frequency of each word
    word_freq = Counter(filtered_words)

    # Select the top N keywords (you can adjust N as needed)
    N = 10
    keywords = word_freq.most_common(N)

    # Print the keywords
    print("Top keywords:")
    for keyword, freq in keywords:
        print(f"{keyword}: {freq}")

    # Save the keywords to a text file
    with open("keywords.txt", "w") as f:
        for keyword, freq in keywords:
            f.write(f"{keyword}\n")
    
    # Extract and summarize keywords
    extracted_keywords = extract_keywords_from_tokens(text, bert_model, bert_tokenizer)
    
    # Print extracted keywords and their summaries
    print("Extracted keywords:")
    for idx, keyword in enumerate(extracted_keywords, start=1):
        print(f"Keyword {idx}: {keyword}")

        # Search the web for the keyword on Wikipedia
        search_url = f"https://en.wikipedia.org/wiki/{keyword}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the relevant information from the search result
        paragraphs = soup.find_all("p")
        extracted_text = ""
        for paragraph in paragraphs:
            extracted_text += paragraph.get_text() + " "
        extracted_text = extracted_text.strip()

        # Generate summary
        summary = generate_summary(extracted_text, bart_model, bart_tokenizer)
        print(f"Summary of {keyword}:")
        print(summary)

def continuous_recording():
    """Record audio in 10-second intervals and process the transcription in real-time."""
    try:
        # Initialize PyAudio
        audio = pyaudio.PyAudio()

        # Check for input devices
        if audio.get_device_count() == 0:
            print("No input device found. Exiting.")
            return

        streams = []

        def record_audio():
            """Record audio continuously in 10-second intervals."""
            while not stop_event.is_set():
                stream = audio.open(format=FORMAT, channels=CHANNELS,
                                    rate=RATE, input=True,
                                    frames_per_buffer=CHUNK)
                streams.append(stream)
                frames = []

                for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK)
                    frames.append(data)

                # Save the recorded data as a WAV file
                with wave.open(OUTPUT_FILENAME, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))

                stream.stop_stream()
                streams.remove(stream)
                stream.close()

        def transcribe_audio():
            """Transcribe the recorded audio in real-time."""
            while not stop_event.is_set():
                if streams:
                    with sr.AudioFile(OUTPUT_FILENAME) as source:
                        audio_data = r.record(source)
                        try:
                            # Recognize the speech using Google Web Speech API
                            text = r.recognize_google(audio_data)
                            print("Transcription: " + text)
                            process_transcription(text)
                        except sr.UnknownValueError:
                            print("Google Speech Recognition could not understand the audio")
                        except sr.RequestError as e:
                            print("Could not request results from Google Speech Recognition service; {0}".format(e))

        # Create and start the recording and transcribing threads
        record_thread = threading.Thread(target=record_audio)
        transcribe_thread = threading.Thread(target=transcribe_audio)
        record_thread.start()
        transcribe_thread.start()

        # Wait for interrupt signal to stop recording
        input("Press Enter to stop recording...\n")
        stop_event.set()

        # Ensure threads complete their work
        record_thread.join()
        transcribe_thread.join()

        # Terminate PyAudio instance
        audio.terminate()

    except OSError as e:
        print(f"OSError encountered: {e}")

# Initialize an event to handle stop signal
stop_event = threading.Event()

# Start continuous recording and transcription
continuous_recording()