import streamlit as st
import requests

st.title('Bienvenue sur l\'API du Projet 4 !')


with st.form(key='my_form'):
	text_input = st.text_input(label='Enter your sentence here:')
	submit_button = st.form_submit_button(label='Get Tags !')

if submit_button:
    response = requests.post("https://vrjr2ghn2hkqghkvf4hmt5.streamlit.app/predict_tags", params={'sentence': text_input})
    try:
        result = response.json()
        st.write(result['response'])
    except requests.exceptions.JSONDecodeError as e:
        result = response
        st.write(result)
