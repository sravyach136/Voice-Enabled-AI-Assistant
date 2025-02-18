#Load credentials
import os
import time
from dotenv import load_dotenv
import librosa
#to record audios
import wave
import pyaudio #audio in real-time
from scipy.io import wavfile #to get audio in .wav file to feed into wishp
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


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
        data = stream.read(4096)
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
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
    return processor, model


# Function to transcribe audio with latency tracking
def transcribe_audio(processor, model, file_path):
    start_time = time.perf_counter()  # Start timer
    audio_input, _ = librosa.load(file_path, sr=16000)
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    stt_time = time.perf_counter() - start_time  # End timer
    print(f"STT Time Taken: {stt_time:.3f} seconds")  # Log latency
    return transcription.strip()
