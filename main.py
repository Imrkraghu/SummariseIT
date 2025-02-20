## this is the main file which is going to execute 
import pyaudio
import wave
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import pandas as pd
import requests
from bs4 import BeautifulSoup
import gradio as gr

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-trained BART model for summarization
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# Load dataset containing valid keywords
datapath = r"C:\Users\vishal pal\Desktop\summariseIT\SummariseIT\dataset.csv"
df = pd.read_csv(datapath)

# Ensure all keywords are lowercase for consistent matching
valid_keywords = set(df.stack().dropna().str.strip().str.lower().tolist())

def transcribe_audio(audio_filename):
    """Convert speech from an audio file to text using Google Speech Recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_filename) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

def extract_keywords(text):
    """Extract valid keywords from transcribed text."""
    words = word_tokenize(text.lower())  # Convert to lowercase for consistency
    stop_words = set(stopwords.words("english"))
    
    # Remove stopwords and select valid keywords
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    found_keywords = [word for word in filtered_words if word in valid_keywords]
    
    return list(set(found_keywords))  # Remove duplicates

def generate_summary(keyword):
    """Fetch Wikipedia content for a keyword and summarize it using BART."""
    search_url = f"https://en.wikipedia.org/wiki/{keyword.replace(' ', '_')}"  # Replace spaces with underscores
    headers = {"User-Agent": "Mozilla/5.0"}
    
    response = requests.get(search_url, headers=headers)
    
    if response.status_code != 200:
        return f"Could not find Wikipedia page for '{keyword}'"

    soup = BeautifulSoup(response.content, "html.parser")
    paragraphs = soup.find_all("p")

    extracted_text = " ".join([p.get_text() for p in paragraphs]).strip()

    if len(extracted_text) == 0:
        return f"No relevant Wikipedia content for '{keyword}'"

    # Tokenize input for BART model
    inputs = bart_tokenizer(extracted_text, return_tensors='pt', padding=True, truncation=True, max_length=1024)
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main_interface(audio_input):
    """Main function for Gradio: Transcribe, extract keywords, and summarize."""
    transcribed_text = transcribe_audio(audio_input)

    if "could not" in transcribed_text.lower():
        return transcribed_text  # Handle transcription errors

    keywords = extract_keywords(transcribed_text)
    
    if not keywords:
        return "No valid keywords found in the transcription."

    summaries = {keyword: generate_summary(keyword) for keyword in keywords}
    
    result = f"**Transcribed Text:**\n{transcribed_text}\n\n"
    result += "**Summaries:**\n"
    for keyword, summary in summaries.items():
        result += f"- **{keyword.capitalize()}**: {summary}\n\n"

    return result

# ✅ Gradio interface
iface = gr.Interface(
    fn=main_interface,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Audio to Wikipedia Summary",
    description="Upload an audio file, extract key topics, and get Wikipedia-based summaries.",
)

# Launch Gradio app
iface.launch()
