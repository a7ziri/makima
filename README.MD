# Голосовой Ассистент Makima

Голосовой ассистент Makima — это проект на Python, который использует технологии преобразования речи в текст (STT), языковую модель (LLM) и преобразование текста в речь (TTS) для взаимодействия с пользователями через голосовые команды.

## Возможности

- **Преобразование речи в текст (STT):** Преобразует устную речь в текст с использованием модели WhisperModel.
- **Языковая модель (LLM):** Обрабатывает ввод пользователя и генерирует ответы с использованием языковой модели OpenAI.
- **Преобразование текста в речь (TTS):** Преобразует текстовые ответы в речь с использованием библиотеки TTS.

## Установка

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/makima-voice-assistant.git
   cd makima-voice-assistant
   ```

2. Установите необходимые пакеты:
   ```bash
   pip install -r requirements.txt
   ```

3. **Настройте конфигурацию:**
   - Создайте файл `config_file.json` в директории `models` со следующей структурой:
     ```json
     {
       "url": "ваш_openai_api_url",
       "key": "ваш_openai_api_key",
       "history": [],
       "model_size": "base",
       "compute_type": "float16"
     }
     ```

4. 

## Использование

1. Запустите приложение:
   ```bash
   python app.py
   ```

2. Взаимодействуйте с Makima:
   - Говорите в микрофон, когда вас попросят.
   - Скажите "exit" или "quit", чтобы завершить сеанс.

## Зависимости

- Python 3.x
- PyTorch
- OpenAI API
- Библиотека TTS
- PyAudio
- Pygame
- Sounddevice
- Faster Whisper

## Вклад

Буду  рад вашему вкладу! Пожалуйста, форкните репозиторий и отправьте pull request для любых улучшений или исправлений ошибок.

## Лицензия

Этот проект лицензирован по лицензии MIT. Подробности смотрите в файле [LICENSE](LICENSE).


