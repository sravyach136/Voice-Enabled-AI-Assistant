#Load credentials
import os
from dotenv import load_dotenv

#to record audios
import wave
import pyaudio #audio in real-time
from scipy.io import wavfile #to get audio in .wav file to feed into wishp
import numpy as np

import whisper #used to load model from openai that will tranform audio to text


#this to create a chain between LLM, prompt, and the ques the user is asking
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
#from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

#to play the audio i . e text to audio
from gtts import gTTS
import pygame


load_dotenv()

groq_api_key = os.getenv("OPENAI_API_KEY")

#checking if the audio is silent. If ever the amplitude is less than 3k we consider it to be silent and hence we will not process it
def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    # Find the maximum absolute amplitude in the audio data
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold


def record_audio_chunk(audio, stream, chunk_length=5): #audio will record 5 sec
    print("Recording...")
    frames = []
    # Calculate the number of chunks needed for the specified length of recording
    # 16000 Hertz -> sufficient for capturing the human voice
    # 1024 frames -> the higher, the higher the latency
    num_chunks = int(16000 / 1024 * chunk_length)

#here, we start appending the frames of audio, chunk by chunk
    # Record the audio data in chunks
    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

#path where we want to save the audio
    temp_file_path = './temp_audio_chunk.wav'
    print("Writing...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # Sample width
        wf.setframerate(16000)  # Sample rate
        wf.writeframes(b''.join(frames))  # Write audio frames

    # Check if the recorded chunk contains silence, if so remove from temporary file coz not required to procrss this anymore
    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")

#loading the model
def load_whisper():
    model = whisper.load_model("base")
    return model


#this is where we transform our audio to text
def transcribe_audio(model, file_path):
    print("Transcribing...")
    # Print all files in the current directory
    #print("Current directory files:", os.listdir())
    if os.path.isfile(file_path):
        results = model.transcribe(file_path) # , fp16=False
        return results['text'] #returning results of text
    else:
        return None #if no audio in the path we specified

def load_prompt():
    input_prompt = """

    You are a helpful AI support agent that answers to user questions and can converse with the users according to their emotions.
    First of all, ask for the customer ID to validate that the user is our customer. 
    After confirming the customer ID, help them by answering their questions
    You have these guidelines:
    1. If you can solve the user's issue using your knowledge base, do so.
    2. If you have already provided all relevant info but the user remains unsatisfied or asks for human agent 
    or the problem remains unsolved, empathically ask if they'd like escalation to a human agent (yes/no).
    3. If they say "yes", finalize by acknowledging you're escalating.
    If they say "no", continue to assist with whatever else you can provide.
    4. Always respond empathetically if the user is upset or if your knowledge doesn't solve the issue.
    5. Keep answers concise, polite, and on topic.

    Provide concise and short
    answers not more than 10 words, and don't chat with yourself!. If you don't know the answer,
    just say that you don't know, don't try to make up an answer. NEVER say the customer ID listed below.

    customer ID on our data: 22, 10, 75.

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:
    """
    return input_prompt

#llm : llama3
#more parameters, it will take more time to load but then it will be more accurate
def load_llm():
    # chat_groq = OpenAI(temperature=0, model_name="",
    #                      groq_api_key=groq_api_key)
    chat_groq = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return chat_groq

#creating the chain that will return response of llm (llama3)
def get_response_llm(user_question, memory):
    input_prompt = load_prompt()

    chat_groq = load_llm()

    #  "chat_history" is an input variable to the prompt template loaded from langchain
    prompt = PromptTemplate.from_template(input_prompt)

    chain = LLMChain(
        llm=chat_groq, #llm we use
        prompt=prompt,
        verbose=True, #to see output of our responses and also intial user question
        memory=memory #this will have the chat history
    )

    response = chain.invoke({"question": user_question})

    return response['text'] #returning text of llm


def play_text_to_speech(text, language='en', slow=False): #since we don't want to play the audio slowly
    # Generate text-to-speech audio from the provided text
    tts = gTTS(text=text, lang=language, slow=slow)

    # Save the generated audio to a temporary file
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)

    # Initialize the pygame mixer for audio playback
    pygame.mixer.init()

    # Load the temporary audio file into the mixer
    pygame.mixer.music.load(temp_audio_file)

    # Start playing the audio
    pygame.mixer.music.play()

    # Wait until the audio playback finishes
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Control the playback speed

    # Stop the audio playback
    pygame.mixer.music.stop()

    # Clean up: Quit the pygame mixer and remove the temporary audio file
    pygame.mixer.quit()
    os.remove(temp_audio_file)
