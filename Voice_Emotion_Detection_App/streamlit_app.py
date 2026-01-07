import streamlit as st
import pickle
import os
import speech_recognition as sr
from deep_translator import GoogleTranslator
from utils.text_preprocessing import clean_text
import warnings

warnings.filterwarnings("ignore")

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Voice Emotion Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# PREMIUM CSS
# ---------------------------
st.markdown("""
<style>
header, footer, #MainMenu {display: none !important;}
.block-container {padding-top: 1rem !important;}

.stApp {
    background: linear-gradient(135deg, #e0e7ff, #fef3c7);
}

/* HEADER */
.header-card {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    color: white;
    padding: 24px;
    border-radius: 20px;
    text-align: center;
    font-size: 28px;
    font-weight: 800;
    max-width: 650px;
    margin: auto;
    box-shadow: 0 12px 30px rgba(37,99,235,0.45);
}

.subtitle {
    margin-top: 6px;
    font-size: 14px;
    color: #dbeafe;
}

/* INPUT CARD */
.card {
    background: linear-gradient(120deg, #f8fafc, #eef2ff);
    padding: 28px;
    border-radius: 20px;
    max-width: 650px;
    margin: 25px auto;
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}

/* RESULT */
.result-box {
    margin: 30px auto;
    padding: 18px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    border-radius: 16px;
    color: white;
    max-width: 650px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Voice_Emotion_Detection_Core", "model")

with open(os.path.join(MODEL_DIR, "linear_svm.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

recognizer = sr.Recognizer()
translator = GoogleTranslator(source="auto", target="en")

# ---------------------------
# HEADER (NO WHITE BLOCK BELOW)
# ---------------------------
st.markdown("""
<div class="header-card">
    🎭 Voice Emotion Detection
    <div class="subtitle">
        Detect emotions from voice or text using Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# INPUT SECTION
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

mode = st.radio(
    "Choose Input Mode",
    ["Text Input", "Voice Input"],
    horizontal=True
)

emotion = None

# TEXT INPUT
if mode == "Text Input":
    text = st.text_area("Enter text (Hindi or English)", height=120)

    if st.button("🔍 Detect Emotion"):
        if text.strip():
            translated = translator.translate(text)
            cleaned = clean_text(translated)
            vec = vectorizer.transform([cleaned])
            emotion = model.predict(vec)[0]
        else:
            st.warning("Please enter some text")

# VOICE INPUT
else:
    st.info("🎙️ Speak clearly after clicking the button")

    if st.button("🎧 Start Voice Detection"):
        with st.spinner("Listening..."):
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = recognizer.listen(source, phrase_time_limit=6)

                # ORIGINAL SPOKEN TEXT
                spoken_text = recognizer.recognize_google(audio)

                # TRANSLATED TEXT
                translated_text = translator.translate(spoken_text)

                # MODEL PIPELINE
                cleaned = clean_text(translated_text)
                vec = vectorizer.transform([cleaned])
                emotion = model.predict(vec)[0]

                # SHOW BOTH TEXTS
                st.success(f"🗣 **You said:** {spoken_text}")
                st.info(f"🌐 **Translated (English):** {translated_text}")

            except sr.UnknownValueError:
                st.error("❌ Could not understand audio")
            except Exception as e:
                st.error(str(e))

# ---------------------------
# EMOTION COLOR LOGIC
# ---------------------------
if emotion:
    emotion_lower = emotion.lower()

    positive = ["happy", "joy", "love", "surprise"]
    negative = ["anger", "sad", "fear", "disgust"]

    if emotion_lower in positive:
        color = "linear-gradient(90deg, #22c55e, #16a34a)"
        emoji = "😊"
    elif emotion_lower in negative:
        color = "linear-gradient(90deg, #ef4444, #b91c1c)"
        emoji = "😡"
    else:
        color = "linear-gradient(90deg, #3b82f6, #1e40af)"
        emoji = "😐"

    st.markdown(
        f"""
        <div class="result-box" style="background:{color}">
            {emoji} Detected Emotion: {emotion.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )
