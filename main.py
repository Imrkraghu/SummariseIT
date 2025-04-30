# import os
# import pyaudio
# import wave
# import speech_recognition as sr
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from collections import Counter
# from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
# import torch
# import pandas as pd
# import requests
# from bs4 import BeautifulSoup
# import gradio as gr
# import pyttsx3
# import re

# # Define custom NLTK data path
# NLTK_CUSTOM_PATH = 'nltk_resources'

# # Ensure the folder exists
# os.makedirs(NLTK_CUSTOM_PATH, exist_ok=True)

# # Add the custom path to NLTK's search locations
# nltk.data.path.append(NLTK_CUSTOM_PATH)

# # Function to check if an NLTK resource exists
# def is_resource_available(resource_path):
#     try:
#         nltk.data.find(resource_path)
#         return True
#     except LookupError:
#         return False

# # Download missing NLTK resources
# for resource in ['punkt', 'punkt_tab', 'stopwords']:
#     if not is_resource_available(f'tokenizers/{resource}'):
#         nltk.download(resource, download_dir=NLTK_CUSTOM_PATH)

# # audio recording parameters removed 

# # Function to record audio and convert it into a text file
# def record_audio():
#     # copied from module 1
#     ### Initializing The Recognizer
#     r = sr.Recognizer()
#     # Parameters
#     FORMAT = pyaudio.paInt16  # Audio format
#     CHANNELS = 1  # Number of channels
#     RATE = 44100  # Sample rate (Hz)
#     CHUNK = 1024  # Chunk size (number of frames per buffer)
#     RECORD_SECONDS = 10  # Duration of the recording (seconds)
#     OUTPUT_FILENAME = "recorded_audio.wav"  # Output filename

#     try:
#         # Initialize PyAudio
#         audio = pyaudio.PyAudio()

#         # Open stream
#         stream = audio.open(format=FORMAT, channels=CHANNELS,
#                             rate=RATE, input=True,
#                             frames_per_buffer=CHUNK)

#         frames = []

#         # Record data
#         for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
#             data = stream.read(CHUNK)
#             frames.append(data)

#         # Stop and close the stream
#         stream.stop_stream()
#         stream.close()
#         audio.terminate()

#         # Save the recorded data as a WAV file
#         with wave.open(OUTPUT_FILENAME, 'wb') as wf:
#             wf.setnchannels(CHANNELS)
#             wf.setsampwidth(audio.get_sample_size(FORMAT))
#             wf.setframerate(RATE)
#             wf.writeframes(b''.join(frames))

#         print(f"Audio recorded and saved as {OUTPUT_FILENAME}")
#         with sr.AudioFile(OUTPUT_FILENAME) as source:
#             audio = r.record(source)  # Read the entire audio file

#             try:
#                 # Recognize the speech using Google Web Speech API
#                 text = r.recognize_google(audio)
#                 print("Transcription: " + text)
#                 # Append the transcription to a text file
#                 with open("transcription.txt", "a") as f:
#                     f.write(text + "\n")
#                 # Save the transcription to a text file
#                 with open("transcription.txt", "w") as f:
#                     f.write(text)
                    
#             except sr.UnknownValueError:
#                 print("Google Speech Recognition could not understand the audio")
#             except sr.RequestError as e:
#                 print("Could not request results from Google Speech Recognition service; {0}".format(e))
#     except OSError as e:
#         print(f"OSError encountered: {e}")
# #speech recognition function was removed and combined above 

# # Extract keywords from text using tokenization 
# #module 2 used here
# def extract_keywords(text, num_keywords=10):
#     """Extracts top keywords from transcribed text using NLTK."""

#     # Read the text file
#     with open("transcription.txt", "r") as file:
#         text = file.read()

#     # Tokenize the text
#     words = word_tokenize(text)

#     # Remove punctuation and make lowercase
#     words = [word.lower() for word in words if word.isalnum()]

#     # Remove stop words
#     stop_words = set(stopwords.words("english"))
#     filtered_words = [word for word in words if word not in stop_words]

#     # Count the frequency of each word
#     word_freq = Counter(filtered_words)

#     # Select the top N keywords (you can adjust N as needed)
#     N = 10
#     keywords = word_freq.most_common(N)

#     # Print the keywords
#     # print("Top keywords:")
#     # for keyword, freq in keywords:
#     #     print(f"{keyword}: {freq}")

#     # Save the keywords to a text file
#     with open("keywords.txt", "w") as file:
#         for keyword, freq in keywords:
#             file.write(f"{keyword}\n")

# # Function to extract keywords from 'transcription.txt' and save them to 'keywords.txt'
# # is removed and merged above 


# # Load pre-trained model and tokenizer
# model = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# # Loading the dataset of keywords
# datapath = r"dataset.csv"
# df = pd.read_csv(datapath)
# # Flatten the dataset to create a set of valid keywords
# valid_keywords = set()
# for column in df.columns:
#     valid_keywords.update(df[column].dropna().str.strip().tolist())

# def extract_keywords_from_tokens(text, model, tokenizer, num_keywords=5):
#     # Tokenize input
#     inputs = tokenizer(text, return_tensors='pt')
    
#     # Get embeddings
#     with torch.no_grad():
#         outputs = model(**inputs)
#         last_hidden_states = outputs.last_hidden_state

#     # Convert token IDs to tokens
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
#     # Get the [CLS] token's embedding
#     cls_embedding = last_hidden_states[:, 0, :].squeeze()
    
#     # Calculate similarity between each token embedding and the [CLS] embedding
#     similarities = torch.matmul(last_hidden_states.squeeze(), cls_embedding)
    
#     # Get the indices of the top-n tokens with the highest similarity
#     top_indices = similarities.topk(num_keywords).indices

#     # Extract the corresponding tokens, excluding [CLS] and checking if they are in valid_keywords
#     keywords = [tokens[i] for i in top_indices if tokens[i] != '[CLS]' and tokens[i] in valid_keywords]
    
#     return keywords

# # Read and process input text
# file_path = "keywords.txt"
# with open(file_path, "r") as file:
#     text = file.read()

# # Use the function to extract keywords
# keywords = extract_keywords_from_tokens(text, model, tokenizer)

# # Print extracted keywords
# print("Extracted keywords:")
# for idx, keyword in enumerate(keywords, start=1):
#     print(f"Keyword {idx}: {keyword}")

# # Summarization using BART
# bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
# bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# def generate_summary(keyword):
#     """Fetches Wikipedia content for a keyword and summarizes it using BART."""
#     search_url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
#     headers = {"User-Agent": "Mozilla/5.0"}
#     response = requests.get(search_url, headers=headers)
    
#     if response.status_code != 200:
#         return f"Could not find Wikipedia page for '{keyword}'"
    
#     soup = BeautifulSoup(response.content, "html.parser")
#     paragraphs = soup.find_all("p")
#     extracted_text = " ".join([p.get_text() for p in paragraphs]).strip()

#     if not extracted_text:
#         return f"No relevant Wikipedia content for '{keyword}'"

#     inputs = bart_tokenizer(extracted_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
#     summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    
#     return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# # Main function to process audio, extract keywords, and generate summaries
# def main_interface(audio_input):
#     """Processes recorded audio: transcribes, extracts keywords, and summarizes them."""
#     transcribed_text = transcribe_audio(audio_input)

#     if "could not" in transcribed_text.lower():
#         return transcribed_text

#     extract_keywords_from_file("transcription.txt", "keywords.txt", num_keywords=10)

#     with open("keywords.txt", "r") as file:
#         keyword_text = file.read().strip()

#     refined_keywords = extract_keywords_with_bert(keyword_text, bert_model, bert_tokenizer)

#     if not refined_keywords:
#         return "No valid keywords found in the transcription."

#     summaries = {keyword: generate_summary(keyword) for keyword in refined_keywords}

#     result = f"**Transcribed Text:**\n{transcribed_text}\n\n**Summaries:**\n"
#     for keyword, summary in summaries.items():
#         result += f"- **{keyword.capitalize()}**: {summary}\n\n"

#     return result

# # ✅ Gradio interface
# iface = gr.Interface(
#     fn=main_interface,
#     inputs=gr.Audio(type="filepath"),
#     outputs="text",
#     title="SummariseIT",
#     description="Record a conversation, extract keywords, and get summaries of the keywords.",
# )

# # Launch Gradio app
# iface.launch()

# this is the new working code of the file only thing is that it doesnot have a User Interface 
import speech_recognition as sr
import pyttsx3
r = sr.Recognizer()
import pyaudio
import wave
import speech_recognition as sr
import pyttsx3
### Initializing The Recognizer
r = sr.Recognizer()
# Parameters
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Number of channels
RATE = 44100  # Sample rate (Hz)
CHUNK = 1024  # Chunk size (number of frames per buffer)
RECORD_SECONDS = 10  # Duration of the recording (seconds)
OUTPUT_FILENAME = "recorded_audio.wav"  # Output filename

try:
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    # Record data
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"Audio recorded and saved as {OUTPUT_FILENAME}")
    with sr.AudioFile(OUTPUT_FILENAME) as source:
        audio = r.record(source)  # Read the entire audio file

        try:
            # Recognize the speech using Google Web Speech API
            text = r.recognize_google(audio)
            print("Transcription: " + text)
            # Append the transcription to a text file
            with open("transcription.txt", "a") as f:
                f.write(text + "\n")
            # Save the transcription to a text file
            with open("transcription.txt", "w") as f:
                f.write(text)
                
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
except OSError as e:
    print(f"OSError encountered: {e}")
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Read the text file
with open("transcription.txt", "r") as file:
    text = file.read()

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
with open("keywords.txt", "w") as file:
    for keyword, freq in keywords:
        file.write(f"{keyword}\n")

from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Read and process input text
file_path = "keywords.txt"
with open(file_path, "r") as file:
    text = file.read()

# Tokenize input
inputs = tokenizer(text, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

print(last_hidden_states)

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel
import requests
from bs4 import BeautifulSoup
import re

# Load pre-trained model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Loading the dataset of keywords
datapath = r"dataset.csv"
df = pd.read_csv(datapath)
# Flatten the dataset to create a set of valid keywords
valid_keywords = set()
for column in df.columns:
    valid_keywords.update(df[column].dropna().str.strip().tolist())

def extract_keywords_from_tokens(text, model, tokenizer, num_keywords=5):
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Get the [CLS] token's embedding
    cls_embedding = last_hidden_states[:, 0, :].squeeze()
    
    # Calculate similarity between each token embedding and the [CLS] embedding
    similarities = torch.matmul(last_hidden_states.squeeze(), cls_embedding)
    
    # Get the indices of the top-n tokens with the highest similarity
    top_indices = similarities.topk(min(num_keywords, len(similarities))).indices

    # Extract the corresponding tokens, excluding [CLS] and checking if they are in valid_keywords
    keywords = [tokens[i] for i in top_indices if tokens[i] != '[CLS]' and tokens[i] in valid_keywords]
    
    return keywords

# Read and process input text
file_path = "keywords.txt"
with open(file_path, "r") as file:
    text = file.read()

# Use the function to extract keywords
keywords = extract_keywords_from_tokens(text, model, tokenizer)

# Print extracted keywords
print("Extracted keywords:")
for idx, keyword in enumerate(keywords, start=1):
    print(f"Keyword {idx}: {keyword}")

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
import requests
from bs4 import BeautifulSoup

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained BART model and tokenizer for summarization
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Loading the dataset of keywords
datapath = r"dataset.csv"
df = pd.read_csv(datapath)
# Flatten the dataset to create a set of valid keywords
valid_keywords = set()
for column in df.columns:
    valid_keywords.update(df[column].dropna().str.strip().tolist())

def extract_keywords_from_tokens(text, model, tokenizer, num_keywords=5):
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

# Read and process input text
file_path = "keywords.txt"
with open(file_path, "r") as file:
    text = file.read()

# Use the function to extract keywords
def extract_keywords_from_tokens(text, model, tokenizer, num_keywords=5):
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
    top_indices = similarities.topk(min(num_keywords, len(similarities))).indices

    # Extract the corresponding tokens, excluding [CLS] and checking if they are in valid_keywords
    keywords = [tokens[i] for i in top_indices if tokens[i] != '[CLS]' and tokens[i] in valid_keywords]
    
    return keywords

# Read and process input text
file_path = "keywords.txt"
with open(file_path, "r") as file:
    text = file.read().strip()

# Use the function to extract keywords
keywords = extract_keywords_from_tokens(text, bert_model, bert_tokenizer)

# Check if keywords were extracted
if keywords:
    print("Extracted keywords:")
    for idx, keyword in enumerate(keywords, start=1):
        print(f"Keyword {idx}: {keyword}")

        # Search the web for the keyword on Wikipedia
        search_url = f"https://en.wikipedia.org/wiki/{keyword}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract the relevant information from the search result
        paragraphs = soup.find_all("p")
        extracted_text = " ".join([p.get_text() for p in paragraphs]).strip()

        # Generate summary using BART
        def generate_summary(text, model, tokenizer):
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
            summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Generate summary
        summary = generate_summary(extracted_text, bart_model, bart_tokenizer)
        print(f"Summary of {keyword}: {summary}")
else:
    print("No keywords found.")