import time
import numpy as np
import pyaudio  # For real-time audio recording
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Check if audio data is silent. If the max amplitude is below the threshold, we consider it silent.
def is_silence(data, max_amplitude_threshold=500):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    #print(f"Max amplitude detected: {max_amplitude}")  # Debug: log max amplitude
    return max_amplitude <= max_amplitude_threshold

# Record audio continuously until 3 seconds of silence is detected.
def record_audio_until_silence(silence_duration_threshold=2.0):
    print("Recording...")
    p = pyaudio.PyAudio()
    # Use a frames_per_buffer of 1024 samples (approx. 0.064 sec per chunk at 16kHz)
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                    input=True, frames_per_buffer=1024)
    frames = []
    silence_time = 0.0
    chunk_duration = 1024 / 16000.0  # Duration (in seconds) of one chunk

    while True:
        data = stream.read(1024)
        frames.append(data)
        # Convert the chunk to a NumPy array (int16)
        chunk_array = np.frombuffer(data, dtype=np.int16)
        # Update silence counter based on whether the chunk is silent
        if is_silence(chunk_array):
            silence_time += chunk_duration
        else:
            silence_time = 0.0  # Reset if speech is detected
        # End recording if continuous silence exceeds the threshold
        if silence_time >= silence_duration_threshold:
            print(f"Detected {silence_time:.2f} seconds of continuous silence, ending recording.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    print("Processing in-memory...")
    raw_data = b''.join(frames)
    audio_np = np.frombuffer(raw_data, dtype=np.int16)
    return audio_np

# Load the Whisper model and processor.
def load_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
    return processor, model

# Transcribe the in-memory audio and measure latency.
def transcribe_audio(processor, model, audio_array):
    start_time = time.perf_counter()
    # Normalize audio (convert int16 to float32 in [-1, 1])
    audio_input = audio_array.astype('float32') / 32768.0
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    stt_time = time.perf_counter() - start_time
    print(f"STT Time Taken: {stt_time:.3f} seconds")
    return transcription.strip()

def main():
    processor, model = load_whisper()
    print("Whisper model loaded.")
    
    while True:
        user_input = input("Press Enter to start recording (or type 'exit' to quit): ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        audio_array = record_audio_until_silence(silence_duration_threshold=2.0)
        if audio_array is None or len(audio_array) == 0:
            print("No audio recorded. Please try again.")
            continue
        
        transcription = transcribe_audio(processor, model, audio_array)
        print("Transcription:", transcription)
    
    print("Conversation ended.")

if __name__ == "__main__":
    main()
