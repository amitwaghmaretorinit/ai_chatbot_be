import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

def normalize_text(text):
    return text.lower().strip()

data = pd.read_csv('chat_data.csv')

data['Question'] = data['Question'].apply(normalize_text)
questions = data['Question'].tolist()
responses = data['Response'].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)

max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(responses)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length))
model.add(LSTM(64, return_sequences=True))
model.add(GlobalAveragePooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # Output layer

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='loss', patience=5)
model.fit(padded_sequences, labels, epochs=100, verbose=1, callbacks=[early_stopping])

def predict_response(user_input):
    normalized_input = normalize_text(user_input)
    sequence = tokenizer.texts_to_sequences([normalized_input])
    padded_input = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_input)
    index = np.argmax(prediction, axis=1)[0]
    return responses[index], label_encoder.classes_[index]  

def start_chat():
    print("Hi! I'm your friendly AI chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response, class_label = predict_response(user_input)
        print(f"Chatbot: {response} (Predicted Class: {class_label})")

if __name__ == "__main__":
    start_chat()
