# Libraries
import streamlit as st
from langchain.llms.openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain.callbacks import get_openai_callback
#from langchain.prompts import PromptTemplate
import os
from function import check_password
import pdf2image
import numpy as np
import cv2
import re
from pytesseract import pytesseract

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain


# PDF split to image and OCR, then return all strings in PDF
def pdf2image_extract_text(_images):
    # Text for pdf extraction
    text = ''

    # Loading bar
    my_bar = st.progress(0, text="Loading...")

    # Loop through images (every page of PDF)
    for index, image in enumerate(_images):
        # Preprocessing image
        ## Convert PIL format to CV2 format
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        ## To grayscale
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        ## Thresholding (image restoration)
        _, preprocessed_image = cv2.threshold(open_cv_image, 170, 255, cv2.THRESH_BINARY)

        # Detect image language
        osd = pytesseract.image_to_osd(preprocessed_image)
        script = re.search("Script: ([a-zA-Z]+)\n", osd).group(1)

        # OCR image
        if script=='Han':
            OCR_text = pytesseract.image_to_string(preprocessed_image, lang='chi_tra')
        elif script=='Japanese':
            OCR_text = pytesseract.image_to_string(preprocessed_image, lang='jpn')
        else:
            OCR_text = pytesseract.image_to_string(preprocessed_image)

        # Store all text after recognition
        text += OCR_text

        # Loading bar update
        my_bar.progress(int(100/len(_images))*(index+1), text=f'Now scanning on page {index+1}')
    
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
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=3000, chunk_overlap=300)
    docs = text_splitter.create_documents([text])

    # Model + Prompt
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", api_key=st.secrets["chatgpt_api"])
    #llm = OpenAI(api_key=st.secrets["chatgpt_api"], temperature=0, model_name="gpt-3.5-turbo")
    #llm = OpenAIChat(model_name="gpt-3.5-turbo", api_key=st.secrets["chatgpt_api"])

    if language=='English':
        map_prompt = """
                        Write a concise summary of the following:
                        "{docs}"
                        CONCISE SUMMARY:
                    """

        reduce_prompt = """
                            Write a concise summary of the following text delimited by triple backquotes.
                            Return your response in bullet points which covers the key points of the text.
                            ```{docs}```
                            BULLET POINT SUMMARY:
                        """

    elif language=='Chinese':
        # Map
        map_template = """寫出以下內容的簡潔摘要:
        "{docs}"
        簡潔總結:"""

        # Reduce
        reduce_template = """以條列式寫出10個簡潔摘要 譬如 1.abc\n2.abc\n
        以涵蓋文字要點的要點形式傳回您的回覆。
        "{docs}"
        重點總結:"""


    # Sub chain+template
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Combination chain
    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=3000, chunk_overlap=0
    )

    # Make split docs
    split_docs = text_splitter.split_documents(docs)

    # Calculate price
    with get_openai_callback() as cb:
        # Summarization
        with st.spinner(f'Summarizing... {pdf.name}'):

            output = map_reduce_chain.invoke(split_docs)['output_text']
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
            ["Chinese", "English", ])
        

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
                    # Notification for user that PDF is image
                    st.warning('This PDF file is "image" format....', icon="✨")

                    # PDF to image
                    _images = pdf2image.convert_from_bytes(pdf.getvalue())

                    # Summarization
                    output = summarization(_images, language)

                # PDF == text, then extract as text
                else:
                    # Notification for user that PDF is image
                    st.warning('This PDF file is "PDF" format....', icon="✨")

                    # Summarization
                    output = summarization(text, language)
                
                #---------------------------------------------------------#


            # Download summary button
            download_output = f'Summary of "{pdf.name}"\n\n' + output
            st.download_button('Download summary!', data=download_output, file_name=f'Summary of _{pdf.name}_.txt')

            # Store summarization history
            with open(os.path.join('history_summarization', f'{pdf.name}_{language}.txt'), 'w') as f:
                f.write(output)