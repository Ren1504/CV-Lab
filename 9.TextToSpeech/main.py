import gtts
from playsound import playsound

text = "Bravo 6 going dark"


tts = gtts.gTTS(text, lang="en")

tts.save("tts.mp3")
playsound("tts.mp3")
