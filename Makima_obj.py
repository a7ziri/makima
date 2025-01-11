from models.llm_model import load_llm_model  
from models.stt_model import load_stt_model  
from models.tts_model import load_tts_model  
from playsound import playsound
import sounddevice as sd
import pyaudio
import numpy as np
import os
import pygame


class Makima_class:
     # Params:
    #   image_data_path - Path to the  image data
    def __init__(self,  image_data_path=None):
   
         
        # models
        self.recognizer = load_stt_model()
        self.client , self.history = load_llm_model()
        self.voice = load_tts_model()


        #global variables
        self.image_data_path = image_data_path
        pygame.mixer.init()


    def get_response_from_llm(self, user_input):
        # Add user input to history before generating the completion
        self.history.append({"role": "user", "content": user_input})

        completion = self.client.chat.completions.create(
            model="model-identifier",
            messages=self.history,
            temperature=0.7,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content

        # Append the assistant's response to the history
        self.history.append(new_message)
        return new_message


    def get_response_from_tts(self, response):
        # Генерируем аудио потоково
        audio_stream = self.voice.tts(
            text=response,
            speaker_wav='static/reference.wav',
            language='ru',
            stream=True
        )

        # Инициализируем pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=1)

        # Создаем очередь для аудио чанков
        audio_queue = pygame.mixer.Channel(0).get_queue()

        for chunk in audio_stream:
            # Преобразуем numpy array в bytes
            chunk_bytes = chunk.tobytes()
            
            # Создаем звуковой объект из чанка
            sound = pygame.mixer.Sound(buffer=chunk_bytes)
            
            # Добавляем звук в очередь
            audio_queue.append(sound)

            # Если это первый чанк, начинаем воспроизведение
            if not pygame.mixer.get_busy():
                pygame.mixer.Channel(0).play(sound)

        # Ждем, пока все аудио не будет воспроизведено
        while pygame.mixer.get_busy():
            pygame.time.Clock().tick(10)

        return True


    def get_response_from_stt(self):
        import time
        import wave
        import audioop
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        WAVE_OUTPUT_FILENAME = "static/voice.wav"
        SILENCE_THRESHOLD = 500  # Adjust this threshold based on your environment
        SILENCE_DURATION = 3  # seconds

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []
        silence_start = None

        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            rms = audioop.rms(data, 2)  # Calculate the root mean square of the audio data
            if rms < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    print("* done recording")
                    break
            else:
                silence_start = None

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        text, info = self.recognizer.transcribe(WAVE_OUTPUT_FILENAME, beam_size=5, language='ru')
        str_to_tts = ''  # Print the transcribed text with timestamps
        for segment in text:
            str_to_tts += segment.text
        return str_to_tts