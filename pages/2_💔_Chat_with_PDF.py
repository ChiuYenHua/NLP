from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
from function import check_password

# Confit
st.set_page_config(page_title='PEACE', page_icon=':earth_asia:', layout='wide')

# Title
st.title('💔 Chat with PDF  (Only Eng version)')

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)
 
load_dotenv()
 
os.environ['OPENAI_API_KEY'] = st.secrets["chatgpt_api"]

# Check password
if check_password():
 
    # upload a PDF file
    pdf = st.file_uploader("", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        # store_name = pdf.name[:-4]
 
        # if os.path.exists(f"data/{store_name}.pkl"):
        #     with open(f"data/{store_name}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
        #     st.write('Embeddings Loaded from the Disk')
        # else:
        embeddings = OpenAIEmbeddings() 
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 

        # Accept user questions/query
        #query = st.text_input("Ask questions about your PDF file:")
 
        
        #st.write(model.config)
        #st.write(VectorStore)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    
        # React to user input
        if prompt := st.chat_input("Ask about PDF !"):


            docs = VectorStore.similarity_search(query=prompt, k=3)
            # st.subheader(":blue[Top 3] most related paragraph based on the question: \n")
            # st.write(docs)
 

            # LLM + QA Chain
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")



            # with get_openai_callback() as cb:
            #     with st.spinner(f'Thinking... \n'):


                    
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})


            # Reponse from LLM
            llm_response = chain.run(input_documents=docs, question=prompt)


            response = f"{llm_response}"
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


            

                    
                    # st.subheader("Answer based on :blue[top 3] most similar paragraph above: \n")
                    # st.info(response, icon="✅")

                    # st.balloons()

                    # st.markdown("""---""") 

                    # st.write(f'Pricing of summary :blue[{pdf.name}]\n')
                    # st.caption(f"Total Tokens: {cb.total_tokens}")
                    # st.caption(f"Prompt Tokens: {cb.prompt_tokens}")
                    # st.caption(f"Completion Tokens: {cb.completion_tokens}")
                    # st.caption(f"Total Cost (USD): ${cb.total_cost}")
 