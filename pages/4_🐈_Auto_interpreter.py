# Libraries
import streamlit as st
import pdf2image
from PIL import Image
from pytesseract import pytesseract
import cv2
import numpy as np
from streamlit_image_comparison import image_comparison

# Confit
st.set_page_config(page_title='PEACE', page_icon=':earth_asia:', layout='wide')

# Title
st.title('🐈 Auto_interpreter')

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

# ----------------------------------------------------------------------------------------------- #


    


