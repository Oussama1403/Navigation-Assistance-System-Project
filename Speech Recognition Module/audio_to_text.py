import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from googlemaps import Client as GoogleMaps
import json


def load_destinations():
    with open("Speech Recognition Module/destinations.json", "r") as f:
        destinations = json.load(f)
        print(destinations)
    return destinations.get("destinations", [])

def process_text(text,destinations):
    tokens = word_tokenize(text) # split text using NTLK
    #if any(word in tokens for word in ["go", "take", "destination"]):
        #print("Navigational command detected.")
    matched_destinations = extract_destinations(tokens,destinations)
    if matched_destinations:
        return matched_destinations
        #generate_route(destination)
    else:
        return "no matched_destinations"

def extract_destinations(tokens,destinations):
    matched_destinations = []
    for destination in destinations:
        if destination.lower() in tokens:
            matched_destinations.append(destination)
    return matched_destinations

def generate_route(destination):
    gmaps = GoogleMaps('<API_KEY>')
    # Example: directions = gmaps.directions("current location", destination, mode="driving")
    print("Generating route to", destination)

def process_speech():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        print("Processing...")
        text = recognizer.recognize_google(audio, language="fr-FR",show_all=True)
        print('\033[91m' , "You said:" , text , '\033[0m')
        with open("Speech Recognition Module/output.txt","w") as f:
            f.write(str(text))
        return text['alternative'][0]['transcript']    
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        print("Error : ", str(e))

def main():
    global recognizer
    recognizer = sr.Recognizer()
    destinations = load_destinations()
    print("destinations : ",destinations)
    while True:
        try:
            text = process_speech()
            matched_destinations = process_text(text.lower(),destinations)
            print('\033[91m' , "found destination " , matched_destinations , '\033[0m')
        except KeyboardInterrupt:
            break
        except:
            print("an error happened, probably not recognized voice")

if __name__ == "__main__":
    main()

