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
    page_title="Voice Emotion AI Pro",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# PREMIUM CSS
# ---------------------------
st.markdown("""
<style>

/* ---------------- BASIC HIDE ---------------- */
header, footer, #MainMenu { display: none !important; }
.block-container { padding-top: 1.5rem !important; }

/* ---------------- APP BACKGROUND (IMAGE REMOVED) ---------------- */
.stApp {
    background: linear-gradient(150deg,#f1dac4, #7c9885);  /*  BG COLOR */
    background-attachment: fixed;
    font-family: 'Segoe UI', Roboto, sans-serif;
}

/* ---------------- HEADER CARD ---------------- */
.header-card {
    background: linear-gradient(90deg, #003049, #084c61);
    color: #eae2b7;
    padding: 26px;
    border-radius: 22px;
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    max-width: 680px;
    margin: 0 auto 25px auto;
    box-shadow: 0 15px 35px rgba(0,48,73,0.45);
}

.subtitle {
    margin-top: 6px;
    font-size: 15px;
    color: #ffc857;
}

/* ---------------- LABEL ---------------- */
.custom-label {
    font-weight: 800;
    color: #003049;
    font-size: 15px;
    margin-bottom: 8px;
}

/* ================= RADIO BUTTONS ================= */
div[data-testid="stRadio"] > div {
    background: linear-gradient(90deg,#eaeaea);
    padding: 10px 18px;
    border-radius: 30px;
    border: 2px solid #f77f00;
    display: flex;
    gap: 30px;
    align-items: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.12);
}

div[data-testid="stRadio"] label {
    font-weight: 700 !important;
    font-size: 15px !important;
    color:   #000000!important;
}

/* ================= TEXT AREA ================= */
textarea {
    background: linear-gradient(100deg,  #e8dab2) !important;
    color: #003049 !important;
    border-radius: 16px !important;
    border: 2px solid #ffc857 !important;
    font-size: 16px !important;
    padding: 12px !important;
    box-shadow: 0 8px 18px rgba(0,0,0,0.12);
}

/* ---------------- INFO CARDS ---------------- */
.info-card {
    padding: 20px;
    border-radius: 18px;
    margin-bottom: 18px;
    font-weight: 600;
    font-size: 18px;
    color: #000000;
}

.speak-hint {
    background: linear-gradient(90deg, #a69cac);
}

.you-said {
    background: linear-gradient(90deg, #caf0f8);
}

.translated {
    background: linear-gradient(90deg, #b5b682);
}

/* ---------------- BUTTON ---------------- */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #1982c4);
    color: #000000;
    border-radius: 15px;
    padding: 12px;
    font-weight: 700;
    border: none;
    box-shadow: 0 10px 25px rgba(0,0,0,0.35);
}

/* ---------------- RESULT BOX ---------------- */
.result-box {
    margin: 30px auto;
    padding: 25px;
    text-align: center;
    font-size: 28px;
    font-weight: 800;
    border-radius: 20px;
    color:   #000000;
    background: linear-gradient(90deg,  #ffca3a );
    box-shadow: 0 12px 30px rgba(0,0,0,0.3);
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD MODEL
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "Voice_Emotion_Detection_Core", "model")

@st.cache_resource
def load_assets():
    try:
        with open(os.path.join(MODEL_DIR, "linear_svm.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except:
        return None, None

model, vectorizer = load_assets()

if model is None or vectorizer is None:
    st.error(" Model files not found. Check model directory.")
    st.stop()

recognizer = sr.Recognizer()
translator = GoogleTranslator(source="auto", target="en")

# ---------------------------
# UI
# ---------------------------
st.markdown("""
<div class="header-card">
    Voice Emotion Detection
    <div class="subtitle">Detect emotions from voice or text using ML</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="custom-label">Select Analysis Mode</p>', unsafe_allow_html=True)
mode = st.radio("", ["Text Input", "Voice Input"], horizontal=True)

final_spoken = ""
final_trans = ""
emotion = None

# ---------------------------
# TEXT MODE
# ---------------------------
if mode == "Text Input":
    text_data = st.text_area("Enter your sentence", height=120)

    if st.button("Run Emotion Analysis"):
        if text_data.strip():
            final_spoken = text_data
            final_trans = translator.translate(text_data)
            cleaned = clean_text(final_trans)
            emotion = model.predict(vectorizer.transform([cleaned]))[0]

# ---------------------------
# VOICE MODE
# ---------------------------
else:
    st.markdown('<div class="info-card speak-hint">Speak clearly after clicking below</div>', unsafe_allow_html=True)

    if st.button("Start Voice Detection"):
        with st.spinner("Listening..."):
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.8)
                    audio = recognizer.listen(source, phrase_time_limit=6)

                final_spoken = recognizer.recognize_google(audio)
                final_trans = translator.translate(final_spoken)
                cleaned = clean_text(final_trans)
                emotion = model.predict(vectorizer.transform([cleaned]))[0]

            except:
                st.error(" Could not process audio")


# ---------------------------
# OUTPUT
# ---------------------------
if final_spoken:
    st.markdown(
        f'<div class="info-card you-said">You said: {final_spoken}</div>',
        unsafe_allow_html=True
    )

if final_trans:
    st.markdown(
        f'<div class="info-card translated">Translated: {final_trans}</div>',
        unsafe_allow_html=True
    )

if emotion:
    st.markdown(
        f'''
        <div class="result-box" style="background: linear-gradient(90deg, #8f2d56);">
            Detected Emotion: {emotion.upper()}
        </div>
        ''',
        unsafe_allow_html=True
    )

