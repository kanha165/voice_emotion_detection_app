import os
import pickle
import speech_recognition as sr
from deep_translator import GoogleTranslator

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR, "..", "Voice_Emotion_Detection_Core", "model"
)

model = pickle.load(
    open(os.path.join(MODEL_DIR, "linear_svm.pkl"), "rb")
)

vectorizer = pickle.load(
    open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb")
)

from utils.text_preprocessing import clean_text

# =========================
# SPEECH SETUP
# =========================
recognizer = sr.Recognizer()
translator = GoogleTranslator(source="auto", target="en")

print(" Speak now (Hindi or English)...")

with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
    audio = recognizer.listen(source, phrase_time_limit=10)

try:
    text = recognizer.recognize_google(audio)
    print("You said:", text)

    translated = translator.translate(text)
    print("Translated:", translated)

    cleaned = clean_text(translated)
    vec = vectorizer.transform([cleaned])

    emotion = model.predict(vec)[0]
    print("Detected Emotion:", emotion)

except sr.UnknownValueError:
    print("Could not understand audio")

except Exception as e:
    print("Error:", e)
