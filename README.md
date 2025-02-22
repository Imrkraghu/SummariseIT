# SummariseIT

SummariseIT takes voice input and creates notes of keywords in real time. This project involves building a voice-to-text converter that takes real-time voice recordings and generates a text file.

## Getting Started

### Prerequisites

- Python 3.10
- All required dependencies are listed in the `requirements.txt` file.

### Installation

1. Clone the repository.
2. **Update the package list**: This ensures your package manager has the most recent list of available packages and their versions.
   ```bash
   sudo apt-get update
   ```
3. **Install PortAudio**: PortAudio is a cross-platform, open-source library that provides audio I/O functionality. PyAudio depends on this library to handle audio streams.
   ```bash
   sudo apt-get install portaudio19-dev
   ```
4. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Install protobuf (if required)**: Protobuf (Protocol Buffers) is a language-neutral, platform-neutral, extensible mechanism for serializing structured data. Some components of the project may rely on this library for data serialization.
   ```bash
   pip install protobuf
   ```

### Usage

1. Execute the `main.py` file to run the initial version. This version takes a manual voice input of a few seconds, which can be configured.
   ```bash
   python main.py
   ```
2. The next version will support fully real-time voice input.

### Project Structure

The project is divided into three modules:

- **module1.ipynb**: Explains the initial setup and basic functionalities.
- **module2.ipynb**: Covers voice-to-text conversion details.
- **module3.ipynb**: Demonstrates how to extract keywords and create notes using the BERT model.

### Files

- **dataset.csv**: Contains the words which can be considered as the keywords to find match during a conversation.
- **recorded_audio.wav**: Contains the recorded audio and is updated each time a new recording is made.
- **transcription.txt**: Stores the transcription of the recorded audio and is updated each time a new recording is made.
- **keywords.txt**: Keywords extracted from the transcription of the voice input which is then used by BERT model to generate summary from wikipedia information.

### User Interface

The main file uses Gradio to run a local web server and provide a user interface for interaction.

---