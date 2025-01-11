import sounddevice as sd
from faster_whisper import WhisperModel
import json

def load_stt_model():
      print('Loading STT model...')
      with open('models/config_file.json', 'r' , encoding='utf-8') as f:
        config = json.load(f)
        model = WhisperModel(config['model_size'] , device="cuda" , compute_type=config['compute_type'])
        print('STT model loaded')
        return model