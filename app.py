import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model, tokenizer, and label encoder
model = load_model('chatbot_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Maximum length of input sequences (should match the training script)
max_length = 20  # Set according to your training script

# Function to predict responses
def predict_response(user_input):
    # Tokenize the user input
    sequence = tokenizer.texts_to_sequences([user_input])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    
    # Predict the response
    predicted_label = model.predict(padded_sequence)
    response_index = predicted_label.argmax(axis=-1)
    
    # Decode the label back to response
    response = label_encoder.inverse_transform(response_index)
    return response[0]


# Define a route for the chatbot
@app.route('/message', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if user_input:
        response = predict_response(user_input)
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'No message provided.'}), 400

if __name__ == "__main__":
    app.run(debug=True)
