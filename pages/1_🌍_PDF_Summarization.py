# Libraries
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
import os
from function import check_password
import pdf2image
import numpy as np
import cv2
from pytesseract import pytesseract


# PDF split to image and OCR, then return all strings in PDF
def pdf2image_extract_text(_images):
    # Notification for user that PDF is image
    st.warning('This PDF file is image format....', icon="✨")

    # Text for pdf extraction
    text = ''

    # Loop through images (every page of PDF)
    for image in _images:
        # Preprocessing image
        ## Convert PIL format to CV2 format
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        ## To grayscale
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        ## Thresholding (image restoration)
        _, preprocessed_image = cv2.threshold(open_cv_image, 170, 255, cv2.THRESH_BINARY)

        # OCR image
        OCR_text = pytesseract.image_to_string(preprocessed_image)

        # Store all text after recognition
        text += OCR_text
    
    return text

# Summarization (split, map_reduce, prompt, langchain_summarization)
def summarization(_image_or_text, language):
    # Determine image or text
    ## Image
    if not isinstance(_image_or_text, str):
        text = pdf2image_extract_text(_image_or_text)
    ## Text
    else:
        text = _image_or_text

    # Replace special character
    text = text.replace('\t', ' ')

    # Split up text
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=2500, chunk_overlap=250)
    docs = text_splitter.create_documents([text])

    # Model + Prompt
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    if language=='English':
        map_prompt = """
                        Write a concise summary of the following:
                        "{text}"
                        CONCISE SUMMARY:
                    """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

        combine_prompt = """
                            Write a concise summary of the following text delimited by triple backquotes.
                            Return your response in bullet points which covers the key points of the text.
                            ```{text}```
                            BULLET POINT SUMMARY:
                        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    elif language=='Chinese':
        map_prompt = """
                        寫出以下內容的簡潔摘要:
                        "{text}"
                        簡潔總結:
                    """
        map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

        combine_prompt = """
                            寫出以下由三重反引號分隔的文字的簡潔摘要。
                            以涵蓋文字要點的要點形式傳回您的回覆。
                            ```{text}```
                            重點總結:
                        """
        combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(llm=llm,
                                        chain_type='map_reduce',
                                        map_prompt=map_prompt_template,
                                        combine_prompt=combine_prompt_template)
 

    # Calculate price
    with get_openai_callback() as cb:
        # Summarization
        with st.spinner(f'Summarizing... {pdf.name}'):

            output = summary_chain.run(docs)
            st.subheader(f'Summary of :blue[{pdf.name}] :sunglasses: \n')
            st.info(output, icon="✅")

            st.balloons()

            st.markdown("""---""") 

            st.write(f'Pricing of summary :blue[{pdf.name}]\n')
            st.caption(f"Total Tokens: {cb.total_tokens}")
            st.caption(f"Prompt Tokens: {cb.prompt_tokens}")
            st.caption(f"Completion Tokens: {cb.completion_tokens}")
            st.caption(f"Total Cost (USD): ${cb.total_cost}")

    return output

# Check is PDF already summarize and stored in folder
def check_pdf_in_history(language):
    for pdf_name in os.listdir('history_summarization'):
        if pdf_name==f'{pdf.name}_{language}.txt':
            return True
        
    return False



# Confit
st.set_page_config(page_title='PEACE', page_icon=':earth_asia:', layout='wide')

# Show history button
show_history = st.toggle('Show summarization history')

# Title
st.title('🌍 PDF Summarization')

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

# Load Chatgpt API key
os.environ['OPENAI_API_KEY'] = st.secrets["chatgpt_api"]


# Check password
if check_password():

    # Toggle to show history or do normal function
    if show_history:
        option = st.selectbox(
                                '',
                                tuple([file_name[:-4] for file_name in os.listdir('history_summarization')]))
        
        f = open(os.path.join('history_summarization', f'{option}.txt'), 'r')
        output = f.read()
        f.close()

        st.subheader(f'Summary of :blue[{option}] :sunglasses: \n')
        st.info(output, icon="✅")

        # Download summary button
        download_output = f'Summary of "{option}"\n\n' + output
        st.download_button('Download summary!', data=download_output, file_name=f'Summary of _{option}_.txt')


    else:
        # Load a PDF file + to text
        pdf = st.file_uploader("", type='pdf')

        # Radio button for choosing 'chinese' or 'english' summarization
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>', unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>', unsafe_allow_html=True)
        language = st.radio(
            "Choose language of summarization !",
            ["English", "Chinese", ])
        

        # PDF uploaded + Summary button = start summary
        if pdf is not None: #and st.button("Start PDF Summarization", type="primary"):

            # if pdf summarization already exist, then read from .txt
            if check_pdf_in_history(language):
                f = open(os.path.join('history_summarization', f'{pdf.name}_{language}.txt'), 'r')
                output = f.read()
                f.close()

                st.subheader(f'Summary of :blue[{pdf.name}] :sunglasses: \n')
                st.info(output, icon="✅")

            # No history, then do summarize
            else:
                # Preprocessing, read PDF
                # Read PDF
                pdf_reader = PdfReader(pdf)

                # String stores all PDF text
                text = ''

                # Scan all pdf text, if PDF is text not image
                for page in pdf_reader.pages:
                    text += page.extract_text()

                #---------------------------------------------------------#

                # PDF == image, then extract as image
                if text=='':
                    # PDF to image
                    _images = pdf2image.convert_from_bytes(pdf.getvalue())

                    # Summarization
                    output = summarization(_images, language)

                # PDF == text, then extract as text
                else:
                    # Summarization
                    output = summarization(text, language)
                
                #---------------------------------------------------------#


            # Download summary button
            download_output = f'Summary of "{pdf.name}"\n\n' + output
            st.download_button('Download summary!', data=download_output, file_name=f'Summary of _{pdf.name}_.txt')

            # Store summarization history
            with open(os.path.join('history_summarization', f'{pdf.name}_{language}.txt'), 'w') as f:
                f.write(output)