import streamlit as st
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore


#load the model for prediction
model = load_model('next_word_predictor.h5')


#load tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer = pickle.load(handle)


#function to predict next word
def predict_next_word(model,tokenizer,text,max_sen_length):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sen_length:
        token_list = token_list[-(max_sen_length-1):]
    token_list = pad_sequences([token_list],maxlen = max_sen_length)
    predicted = model.predict(token_list,verbose =0)
    predicted_word_index  = np.argmax(predicted,axis =1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

    #streamlit app functionality
st.title("Next word predictor using LSTM")
input_text = st.text_input("enter the sequence of words","to be or not to be")
if st.button("predict next word"):
    max_sequence_len = model.input_shape[1]
    word = predict_next_word(model, tokenizer,input_text,max_sequence_len)
    st.write(f'Nxt word :{word}')
