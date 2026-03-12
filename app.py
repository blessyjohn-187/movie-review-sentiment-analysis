import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load model
model = tf.keras.models.load_model("sentiment_model.h5")

# load tokenizer
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

st.title("🎬 Movie Review Sentiment Analyzer")

st.write("Enter a movie review to predict whether it is Positive or Negative.")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):

    seq = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(seq,maxlen=200)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        st.success("😊 Positive Review")

    else:
        st.error("😠 Negative Review")