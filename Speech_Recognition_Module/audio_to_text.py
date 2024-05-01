import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
import json
import os,sys

# Add the parent directory of the current script (Speech_Recognition_Module) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from Navigation_Module import main


def load_destinations():
    with open("Speech_Recognition_Module/destinations.json", "r") as f:
        destinations = json.load(f)
        print(destinations)
    return destinations.get("destinations", [])

def process_text(text,destinations):
    tokens = word_tokenize(text) # split text using NTLK
    print("tokens :",tokens)
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

def process_speech():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.listen(source)
    try:
        print("Processing...")
        text = recognizer.recognize_google(audio, language="fr-FR",show_all=True)
        print('\033[91m' , "You said:" , text , '\033[0m')
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
    
    #while True:
    #    try:
    #          
    #        #nav(matched_destinations)
    #    except KeyboardInterrupt:
    #        break
    #    except:
    #        print("an error happened, probably not recognized voice")
    
    text = process_speech()
    matched_destinations = process_text(text.lower(),destinations)
    print('\033[91m' , "found destination " , matched_destinations , '\033[0m')
    print(matched_destinations)       

if __name__ == "__main__":
    main()

