
import streamlit as st
import joblib

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧"
)

st.title("📧 Email Spam Classifier")
st.write("Enter an email or message to check whether it is Spam or Not Spam.")

st.divider()

message = st.text_area(
    "✉️ Enter your email/message",
    height=180,
    placeholder="Example: Congratulations! You have won a free prize..."
)

if st.button("🔍 Predict", use_container_width=True):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        message_tfidf = vectorizer.transform([message])

        prediction = model.predict(message_tfidf)[0]
        probabilities = model.predict_proba(message_tfidf)[0]

        confidence = max(probabilities) * 100

        st.divider()

        if prediction == 1:
            st.error("🚨 SPAM")
        else:
            st.success("✅ NOT SPAM")

        st.write(f"**Confidence: {confidence:.2f}%**")

st.divider()

st.subheader("🧪 Try a Sample")

st.write("**Spam example:**")
st.code("Congratulations! You have won $100000. Click here to claim your prize!")

st.write("**Normal example:**")
st.code("Hi, please send me the project report before tomorrow's meeting.")
