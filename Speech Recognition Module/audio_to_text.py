import speech_recognition as sr
#import nltk
#from nltk.tokenize import word_tokenize

recognizer = sr.Recognizer()

def process_text(text):
    pass
    # Tokenize the recognized text (split text)
    #tokens = word_tokenize(text)
    # Perform NLP tasks
    # (soon will be implemented)

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
        with open("output.txt","a") as f:
            f.write(text)
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand the audio.")
    except sr.RequestError as e:
        print("Error : ", str(e))


# Main loop for continuous speech recognition
while True:
    process_speech()
