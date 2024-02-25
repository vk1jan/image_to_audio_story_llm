import os
import requests
# from dotenv import load_dotenv,find_env
from transformers import pipeline
from IPython.display import Audio
import streamlit as st
from dotenv import load_dotenv,find_dotenv

load_dotenv(find_dotenv())
hg_api_key = os.getenv("HUGGINGFACE_ACCESS_TOKEN")
# print(f"API KEY==={hg_api_key}")
API_URL_text_2_story = "https://api-inference.huggingface.co/models/jcpwfloi/gpt2-story-generation"
API_URL_text_2_speech = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
headers = {"Authorization": f"Bearer {hg_api_key}"}

def img_to_text_llm(url):
    img2textmodel = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = img2textmodel(url)
    return text[0]["generated_text"]

def query(payload):
	response = requests.post(API_URL_text_2_story, headers=headers, json=payload)
	return response.json()

def speech_query(payload):
	response = requests.post(API_URL_text_2_speech, headers=headers, json=payload)
	return response.content

def text_to_story_llm(text):
	
    output = query({
	    "inputs": text,
        "max_new_tokens":10000,
    })
    print(output)
    return output[0]["generated_text"]

def text_to_speech_llm(story):
    audio = speech_query({
	    "inputs": story,
    })

    with open('audio.mp3','wb') as file:
        file.write(audio)



def main():
    st.set_page_config(page_title="image to audio story")
    st.header("Turn your image into an audio story")
    uploaded_file = st.file_uploader("Choose an image to turn into your personal audio story",type="jpeg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file,caption="Uploaded Image",use_column_width=True)
        text = img_to_text_llm(uploaded_file.name)
        story = text_to_story_llm(text)
        text_to_speech_llm(story)

        with st.expander("Description"):
            st.write(text)

        with st.expander("Story"):
            st.write(story)

        st.audio("audio.mp3")

if __name__ == "__main__":
    main()


