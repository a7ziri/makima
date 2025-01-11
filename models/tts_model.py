import torch
from TTS.api import TTS


def load_tts_model():
    print('Loading TTS model...')
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    print('TTS model loaded')
    return tts