import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load model
model = tf.keras.models.load_model("sentiment_model.h5")

# load tokenizer
with open("tokenizer.pkl","rb") as f:
    tokenizer = pickle.load(f)

print("Sentiment Analysis System")

while True:

    text = input("Enter review (type exit to stop): ")

    if text == "exit":
        break

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq,maxlen=200)

    pred = model.predict(padded)[0][0]

    if pred > 0.5:
        print("Sentiment: Positive 😊")
    else:
        print("Sentiment: Negative 😠")