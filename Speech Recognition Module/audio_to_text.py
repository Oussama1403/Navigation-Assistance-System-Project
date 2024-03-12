import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from googlemaps import Client as GoogleMaps


def process_text(text):
    # split text using NTLK
    tokens = word_tokenize(text)
    #if any(word in tokens for word in ["go", "take", "destination"]):
        #print("Navigational command detected.")
    # Extract destination from text (you can implement more sophisticated NLP here)
    destination = extract_destination(tokens)
    if destination:
        print("Destination:", destination)
        # Call function to generate route using destination
        #generate_route(destination)
    else:
        print("No destination found.")

def extract_destination(tokens):
    # Initialize NER tagger (NER is a type of NLP)
    ner_tagger = nltk.ne_chunk(nltk.pos_tag(tokens))
    destination = None
    for chunk in ner_tagger:
        if hasattr(chunk, 'label') and chunk.label() == 'GPE':
            destination = ' '.join(c[0] for c in chunk.leaves())
            break
    return destination

def generate_route(destination):
    gmaps = GoogleMaps('<API_KEY>')
    # Example: directions = gmaps.directions("current location", destination, mode="driving")
    print("Generating route to", destination)

def process_speech():
    # Record audio from the microphone
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    # Perform speech recognition
    try:
        print("Processing...")
        text = recognizer.recognize_google(audio, language="fr-FR")
        print('\033[91m' + "You said:", text + '\033[0m')
        with open("Speech Recognition Module/output.txt","a") as f:
            f.write(text)
        return text    
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        print("Error : ", str(e))

def main():
    global recognizer
    recognizer = sr.Recognizer()
    while True:
        text = process_speech()
        #process_text(text)


if __name__ == "__main__":
    main()

