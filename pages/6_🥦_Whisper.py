# Libraries
import streamlit as st
from pydub import AudioSegment
import openai
from datetime import timedelta
import os
import re
from function import check_password

# Confit
st.set_page_config(page_title='PEACE', page_icon=':earth_asia:', layout='wide')

# Title
st.title('🥦 Whisper')

# Style
with open('style.css')as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)

#-------------------------------------------------------------------------------#

# Load Chatgpt API key
os.environ['OPENAI_API_KEY'] = st.secrets["chatgpt_api"]

# Check password
if check_password():

    # Upload audio
    uploaded_audio = st.file_uploader("", type=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'])


    if uploaded_audio:
        # Finde audio extension
        extension = str(uploaded_audio.name)[re.search(r'\w*.\w$', str(uploaded_audio.name)).start():]

        # 讀取下載下來的音檔
        audio_file = AudioSegment.from_file(uploaded_audio)

        with st.spinner(f'Cut into chunk...'):
            # 切割音檔成多個小檔案
            chunk_size = 40 * 60 * 1000  # 40分鐘
            chunks = [audio_file[i:i+chunk_size] for i in range(0, len(audio_file), chunk_size)]

        with st.spinner(f'Speech to text...'):
            # 使用 OpenAI 的 Audio API 將每個小檔案轉成文字，然後合併在一起
            transcript = ""
            for chunk in chunks:
                with chunk.export(f"temp.mp3", format='mp3') as f:
                    result = openai.Audio.transcribe("whisper-1", f)
                    transcript += result["text"]

        # 依照我們指定的長度分割字串
        def split_text(text, max_length):
            return [text[i:i+max_length] for i in range(0, len(text), max_length)]

        # 依照 type 處理文字
        def process_text(text, type):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": type + text}
                ]
            )
            return response.choices[0].message.content

        # 處理長字串
        def process_long_text(long_text, type):
            text_list = split_text(long_text, 1200)
            processed_text_list = [process_text(text, type) for text in text_list]
            return "".join(processed_text_list)


        with st.spinner(f'Get comprehensive content...'):
            # 呼叫分段處理函式
            processed_transcript = process_long_text(transcript, "使用繁體中文閱讀，幫我改錯字、加標點符號，並分段使內容更通順：\n")

        with st.spinner(f'Get summarized content...'):
            # 呼叫取得摘要函式
            processed_summary = process_long_text(processed_transcript, "閱讀以下文字，用「-」作為前綴條列出重點，用「繁體中文」呈現：\n")


        st.subheader('Context:')
        st.write(processed_transcript)

        st.write('---')

        st.subheader('Summary:')
        st.write(processed_summary)


        # Store Context + Summary
        with open(os.path.join('history_audio_paragraph', f'{uploaded_audio.name}.txt'), 'w') as f:
            f.write(processed_transcript)
        with open(os.path.join('history_audio_summary', f'{uploaded_audio.name}.txt'), 'w') as f:
            f.write(processed_summary)



        # Statictics
        duration = str(timedelta(seconds=len(audio_file)/1000)).split(':')

        st.caption('Time: {} hour  {} min  {} sec'.format(duration[0], duration[1], duration[2]))
        st.caption(f'Price: ${0.006*(len(audio_file)/1000/60)}')


