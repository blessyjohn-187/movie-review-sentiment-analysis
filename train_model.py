import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# dataset location
train_dir = "aclImdb/train"

texts = []
labels = []

# positive reviews
for file in os.listdir(os.path.join(train_dir,"pos"))[:5000]:
    with open(os.path.join(train_dir,"pos",file),encoding="utf8") as f:
        texts.append(f.read())
        labels.append(1)

# negative reviews
for file in os.listdir(os.path.join(train_dir,"neg"))[:5000]:
    with open(os.path.join(train_dir,"neg",file),encoding="utf8") as f:
        texts.append(f.read())
        labels.append(0)

# tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences,maxlen=200)
y = np.array(labels)

# split data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# model
model = Sequential()
model.add(Embedding(5000,128,input_length=200))
model.add(LSTM(64))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# training
model.fit(X_train,y_train,epochs=3,batch_size=32)

# testing
loss,accuracy = model.evaluate(X_test,y_test)
print("Test Accuracy:",accuracy)

# save model
model.save("sentiment_model.h5")

# save tokenizer
with open("tokenizer.pkl","wb") as f:
    pickle.dump(tokenizer,f)

print("Training complete. Model saved.")