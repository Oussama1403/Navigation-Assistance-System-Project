import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize

def process_text(text):
    # split text using NTLK
    tokens = word_tokenize(text)
    if any(word in tokens for word in ["go", "take", "destination"]):
        print("Navigational command detected.")
        # Extract destination from text (you can implement more sophisticated NLP here)
        destination = extract_destination(tokens)
        if destination:
            print("Destination:", destination)
        else:
            print("No destination found.")

def extract_destination(tokens):
    destinations = ["New York", "Los Angeles", "Chicago"]  # Example list of destinations
    for word in tokens:
        if word in destinations:
            return word
    return None

def process_speech():
    # Record audio from the microphone
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    # Perform speech recognition
    try:
        print("Processing...")
        text = recognizer.recognize_google(audio)
        print('\033[91m' + "You said:", text + '\033[0m')
        with open("Speech Recognition Module/output.txt","a") as f:
            f.write(text)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        print("Error : ", str(e))

def main():
    global recognizer
    recognizer = sr.Recognizer()
    while True:
        process_speech()

if __name__ == "__main__":
    main()

