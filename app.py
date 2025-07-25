import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences   
import pickle
import numpy as np

# Load the model and tokenizer
model = load_model('next_word_predictor.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text,max_sequence_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_length - 1:
        token_list = token_list[-(max_sequence_length - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=-1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# Streamlit app
st.title("Next Word Predictor")
input_text = st.text_input("Enter a sentence:", "to be or not to be ")
if st.button("Predict Next Word"):
    if input_text:
        max_sequence_length = model.input_shape[1] + 1
        predicted_word = predict_next_word(model, tokenizer, input_text, max_sequence_length)
        if predicted_word:
            st.success(f"Predicted next word: {predicted_word}")
        else:
            st.error("Could not predict the next word.")
    else:
        st.warning("Please enter a sentence to predict the next word.")