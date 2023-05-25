import speech_recognition as sr


recognizer = sr.Recognizer()

with sr.Microphone() as source:
    print("Speak something...")
    audio = recognizer.listen(source)
try:
    text = recognizer.recognize_google(audio)    
    print("You said:", text)
except sr.UnknownValueError:
    print("Speech recognition could not understand audio.")
