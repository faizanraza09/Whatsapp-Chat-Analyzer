'''
Name(s): Fatema AlMarzooqi, Faizan Raza, Khadiya Khalid
NetID: faa7626, mr5985, kk4597
Course: Language of Computers CADT-UH 1013EQ
Description: This is a Streamlit app that allows users to upload a WhatsApp chat text file and then displays a wireframe of the parsed chat data
Date: 21/04/2024
'''

import streamlit as st
from parse_whatsapp import parse_whatsapp_chat

st.title("WhatsApp Chat Analyzer")
uploaded_file = st.file_uploader("Upload your WhatsApp chat text file", type="txt")
if uploaded_file is not None:
    file_content = uploaded_file.getvalue().decode("utf-8")
    chat_df = parse_whatsapp_chat(file_content)
    st.write("Parsed Chat Data:")
    st.dataframe(chat_df)
