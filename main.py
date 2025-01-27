import speech_recognition as sr
import pyttsx3
import webbrowser
import os
import datetime
import wikipedia
import random
import subprocess
import json
import requests

# Initialize speech recognition and text-to-speech engines
r = sr.Recognizer()
engine = pyttsx3.init()

# Set voice (optional, depends on your system's available voices)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Use the first voice

def speak(text):
    engine.say(text)
    engine.runAndWait()

def wish_me():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good Morning!")
    elif 12 <= hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am your desktop AI. How can I help you?")

def take_command():
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1  # Adjust for silence between words
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")
    except sr.UnknownValueError:
        print("Could not understand audio")
        return "None"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return "None"
    return query

if __name__ == "__main__":
    wish_me()
    while True:
        query = take_command().lower()

        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            try:
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)
            except wikipedia.exceptions.PageError:
                speak("Page not found on Wikipedia")
            except wikipedia.exceptions.DisambiguationError as e:
                speak("Multiple results found. Please be more specific.")
                print(e.options)
        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
        elif 'open google' in query:
            webbrowser.open("google.com")
        elif 'open stackoverflow' in query:
            webbrowser.open("stackoverflow.com")
        elif 'play music' in query:
            music_dir = 'C:\\path\\to\\your\\music' # Replace with your music directory
            songs = os.listdir(music_dir)
            if songs:
                os.startfile(os.path.join(music_dir, random.choice(songs)))
            else:
                speak("No music files found in the specified directory.")
        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            speak(f"The time is {strTime}")
        elif 'open code' in query:
            codePath = "C:\\Users\\YourUserName\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe" # Replace with your VS Code path
            os.startfile(codePath)
        elif 'search' in query:
            query = query.replace("search", "")
            webbrowser.open(f"https://www.google.com/search?q={query}")
        elif 'weather' in query:
            try:
                api_key = "f2958bb99f8564e6e69a22f741247db89887f7530715cba1bb0ee29e2220787b" # Get your API key from openweathermap.org
                base_url = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
                city_name = query.split("weather in ")[1]
                complete_url = base_url + "appid=" + api_key + "&q=" + city_name
                response = requests.get(complete_url)
                x = response.json()
                if x["cod"] != "404":
                    y = x["main"]
                    current_temperature = y["temp"] - 273.15
                    current_humidity = y["humidity"]
                    z = x["weather"]
                    weather_description = z[0]["description"]
                    speak(f"The temperature in {city_name} is {current_temperature:.2f} degrees Celsius.")
                    speak(f"The humidity is {current_humidity} percent.")
                    speak(f"The weather is described as {weather_description}.")
                else:
                    speak("City not found.")
            except Exception as e:
                print(f"Error fetching weather data: {e}")
                speak("Could not retrieve weather information.")
        elif 'exit' in query or 'quit' in query or 'stop' in query:
            speak("Goodbye!")
            break
        elif query != "None":
            speak("I did not understand that.")