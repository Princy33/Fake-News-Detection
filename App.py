import streamlit as st
import joblib
from transformers import pipeline


# Load Classifier

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")


# Load Generator

generator = pipeline("text-generation", model="distilgpt2")


# Page Config

st.set_page_config(page_title="Ethical Fake News GenAI", layout="wide")

st.title("📰 Fake News Headline Generator & Detector")
st.markdown("### Ethical Generative AI Demonstration")



st.header("🔮 Generate a Headline")

topic = st.text_input("Enter Topic")
tone = st.selectbox("Select Tone", ["Neutral", "Sensational"])

if st.button("Generate Headline"):

    if topic.strip() == "":
        st.warning("Please enter a topic.")
    else:

        if tone == "Sensational":
            prompt = f"Shocking breaking news about {topic}:"
        else:
            prompt = f"News update about {topic}:"

        result = generator(
            prompt,
            max_length=25,
            temperature=0.9,
            num_return_sequences=1
        )

        headline = result[0]["generated_text"]

        st.success("Generated Headline:")
        st.write(headline)

        st.warning("⚠️ This headline is AI-generated and may not be factual.")

# SECTION 2 — CLASSIFIER


st.header("🛡 Detect Fake or Real News")

news_input = st.text_area("Paste headline or article here")

if st.button("Predict"):

    if news_input.strip() == "":
        st.warning("Please enter text.")
    else:
        X = vectorizer.transform([news_input])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        fake_prob = probabilities[0] * 100
        real_prob = probabilities[1] * 100

        if prediction == 1:
            st.success(f"Prediction: REAL ({real_prob:.2f}%)")
        else:
            st.error(f"Prediction: FAKE ({fake_prob:.2f}%)")


# SECTION 3 — ETHICS & SAFEGUARDS


st.header("⚖️ Ethics & Responsible AI")

st.markdown("""
### Risks:
- AI can generate misinformation at scale.
- Sensational tone increases viral spread.
- Can manipulate public opinion.

### Safeguards Implemented:
- Warning labels on generated content.
- Confidence score transparency.
- Separate detection system.
- Educational purpose disclaimer.
- No automatic publishing.
""")
