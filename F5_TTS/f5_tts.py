#!/usr/bin/env python3
"""
f5_tts.py

Usage:
    -conda create -n f5-tts python=3.10
    -conda activate f5-tts
    -pip install git+https://github.com/SWivid/F5-TTS.git
    -python f5_tts.py 

"""

import os
import sys
import subprocess
import requests

# ------------------------------------------------------------------------------
# CONFIGURABLES
# ------------------------------------------------------------------------------
# Example reference audio + model checkpoint
CKPT_URL = "https://huggingface.co/SWivid/F5-TTS/resolve/main/F5TTS_Base/model_1200000.pt"
LOCAL_CKPT_DIR = "ckpts/F5TTS_Base"
LOCAL_CKPT_FILE = os.path.join(LOCAL_CKPT_DIR, "model_1200000.pt")

REFERENCE_AUDIO = "sophia-female.wav"
GEN_TEXT = (
    "The transcriptions API takes as input the audio file you want to transcribe "
    "and the desired output file format for the transcription of the audio. "
    "We currently support multiple input and output file formats."
)

# Where output file is saved
OUTPUT_DIR = "tests"
OUTPUT_NAME = "infer_cli_basic.wav"

# ------------------------------------------------------------------------------
# 1) (Optional) Download the Checkpoint if Needed
# ------------------------------------------------------------------------------
def download_checkpoint():
    """Download the model checkpoint locally if not present."""
    os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
    if not os.path.exists(LOCAL_CKPT_FILE):
        print(f"Downloading checkpoint to: {LOCAL_CKPT_FILE}")
        r = requests.get(CKPT_URL, stream=True)
        with open(LOCAL_CKPT_FILE, 'wb') as f:
            f.write(r.content)
    else:
        print("Checkpoint file already exists, skipping download.")

# ------------------------------------------------------------------------------
# 2) Run Inference Using Installed F5-TTS
# ------------------------------------------------------------------------------
def run_inference():
    """
    Calls the installed F5-TTS module:
      python -m f5tts.infer.infer_cli
    (Adjust 'f5tts' if the actual name is different.)
    """
    REPO_DIR = "F5-TTS"
    infer_script = os.path.join(REPO_DIR, "src", "f5_tts", "infer", "infer_cli.py")

    cmd = [
        sys.executable, infer_script,
        "--model", "F5-TTS",
        "--ckpt_file", LOCAL_CKPT_FILE,
        "--ref_audio", REFERENCE_AUDIO,
        "--ref_text", "",
        "--gen_text", GEN_TEXT
    ]
    subprocess.run(cmd, check=True)



# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    # (Optional) Download checkpoint if needed
    download_checkpoint()

    # Run inference
    run_inference()

    print(
        f"\nInference complete. Check the generated WAV file:\n"
        f"  {os.path.join(OUTPUT_DIR, OUTPUT_NAME)}"
    )

if __name__ == "__main__":
    main()