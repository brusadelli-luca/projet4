import streamlit as st
import requests

st.title('Bienvenue sur l\'API du Projet 4 !')


with st.form(key='my_form'):
	text_input = st.text_input(label='Enter your sentence here:')
	submit_button = st.form_submit_button(label='Get Tags !')

if submit_button:
    response = requests.post("http://localhost:5000/predict_tags", params={'sentence': text_input})
    result = response.json()

    st.write(result['response'])
