# Libraries
import streamlit as st

# Confit
st.set_page_config(page_title='PEACE', page_icon=':earth_asia:', layout='wide')

# Title
st.title('🐈 Auto_interpreter')

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

# ----------------------------------------------------------------------------------------------- #


    


