from openai import OpenAI
import  torch 
import json


def load_llm_model():
    print('Loading LLM model...')
    with open('models\config_file.json', 'r' , encoding='utf-8') as f:
        config = json.load(f)

        client = OpenAI(base_url=config['url'], api_key=config['key'])
        history = config['history'] 
        print('LLM model loaded')
        return client, history