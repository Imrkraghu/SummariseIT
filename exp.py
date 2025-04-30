import os
import pyaudio
import wave
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import BertTokenizer, BertModel, BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
import gradio as gr
import pyttsx3
import re

# Define custom NLTK data path
NLTK_CUSTOM_PATH = 'nltk_resources'

# Ensure the folder exists
os.makedirs(NLTK_CUSTOM_PATH, exist_ok=True)

# Add the custom path to NLTK's search locations
nltk.data.path.append(NLTK_CUSTOM_PATH)

# Function to check if an NLTK resource exists
def is_resource_available(resource_path):
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        return False

# Download missing NLTK resources
for resource in ['punkt', 'stopwords']:
    if not is_resource_available(f'tokenizers/{resource}'):
        nltk.download(resource, download_dir=NLTK_CUSTOM_PATH)

# Function to record audio
def record_audio(output_filename="recorded_audio.wav", duration=10, sample_rate=44100):
    """Record audio and save to a WAV file."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)

    frames = []
    print("Recording...")

    for _ in range(int(sample_rate / 1024 * duration)):
        frames.append(stream.read(1024))

    print("Recording complete.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    return output_filename

# Function to transcribe recorded audio
def transcribe_audio(audio_filename):
    """Convert speech to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()

    if not os.path.exists(audio_filename):
        return "Invalid audio file path."

    with sr.AudioFile(audio_filename) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)

            with open("transcription.txt", "w") as f:
                f.write(transcription)

            return transcription
        except sr.UnknownValueError:
            return "Could not understand audio."
        except sr.RequestError as e:
            return f"Request error: {e}"

# Function to extract keywords from transcribed text
def extract_keywords_from_file(input_file, output_file, num_keywords=10):
    """Extracts keywords from a text file."""
    with open(input_file, "r") as file:
        text = file.read()

    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum() and word not in stopwords.words("english")]

    word_freq = Counter(words)
    keywords = [word for word, _ in word_freq.most_common(num_keywords)]

    with open(output_file, "w") as file:
        file.write("\n".join(keywords))

    return keywords

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to extract refined keywords using BERT
def extract_keywords_with_bert(text, model, tokenizer, num_keywords=5):
    """Extracts keywords using BERT embeddings."""
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    similarities = torch.matmul(last_hidden_states.squeeze(), last_hidden_states[:, 0, :].squeeze())
    
    top_indices = similarities.topk(num_keywords).indices
    keywords = [tokens[i] for i in top_indices if tokens[i].isalpha()]
    
    return keywords

# Function to summarize extracted keywords
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def generate_summary(keyword):
    """Fetches Wikipedia content for a keyword and summarizes it using BART."""
    search_url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return f"Could not find Wikipedia page for '{keyword}'"
    
    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")
    extracted_text = " ".join([p.get_text() for p in paragraphs]).strip()

    if not extracted_text:
        return f"No relevant Wikipedia content for '{keyword}'"

    inputs = bart_tokenizer(extracted_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Main function to record audio, extract keywords, and generate summaries
def main_interface():
    """Runs full process: recording, transcribing, extracting keywords, summarizing."""
    
    audio_path = record_audio()
    
    transcribed_text = transcribe_audio(audio_path)

    if "could not" in transcribed_text.lower():
        return transcribed_text

    extract_keywords_from_file("transcription.txt", "keywords.txt", num_keywords=10)

    with open("keywords.txt", "r") as file:
        keyword_text = file.read().strip()

    refined_keywords = extract_keywords_with_bert(keyword_text, bert_model, bert_tokenizer)

    if not refined_keywords:
        return "No valid keywords found."

    summaries = {keyword: generate_summary(keyword) for keyword in refined_keywords}

    result = f"**Transcribed Text:**\n{transcribed_text}\n\n**Summaries:**\n"
    for keyword, summary in summaries.items():
        result += f"- **{keyword.capitalize()}**: {summary}\n\n"

    return result

# ✅ Gradio Interface
iface = gr.Interface(fn=main_interface, inputs=None, outputs="text", title="SummariseIT", description="Records, transcribes, extracts keywords, and summarizes.")

iface.launch()