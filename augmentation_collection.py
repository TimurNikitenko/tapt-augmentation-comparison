import random
from openai import OpenAI
from cerebras.cloud.sdk import Cerebras
import httpx
import json
import time
import datetime
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from prompts import generate_event_prompt, generate_listing_prompt, generate_junk_prompt


load_dotenv()

CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
proxy = os.getenv("PROXY")

schema = {
        "type": "object",
        "properties": {
            "thought_process": {
                "type": "string",
                "description": "Рассуждения модели (Chain of Thought)"
            },
            "text": {
                "type": "string",
                "description": "Итоговый сгенерированный текст"
            }
        },
        "required": ["thought_process", "text"],
        "additionalProperties": False
    }

model_weights = {
    "cerebras/qwen-3-235b-a22b-instruct-2507": 0.20,
    "deepseek/deepseek-v3.2": 0.15,
    "nvidia/nemotron-3-super-120b-a12b:free": 0.15,
    "openai/gpt-oss-120b": 0.15,
    "meta-llama/llama-3.3-70b-instruct": 0.10,
    "google/gemma-4-31b-it": 0.10,
    "minimax/minimax-m2.7": 0.10,
    "google/gemini-3-flash-preview": 0.05,

}

cerebras_model = "qwen-3-235b-a22b-instruct-2507"


def generate_text_by_openrouter(prompt: str, model: str, temperature: float = 0.8) -> dict:
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        with httpx.Client(proxy=proxy) as http_client:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                http_client=http_client,
                max_retries=1
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "augmentation_schema",
                        "strict": True,
                        "schema": schema,
                    },
                },
                top_p=0.9
            )
            
            raw_content = response.choices[0].message.content
            result_dict = json.loads(raw_content)
            
            usage = response.usage
            result_dict['tokens'] = {
                'prompt_tokens': usage.prompt_tokens if usage else 0,
                'completion_tokens': usage.completion_tokens if usage else 0,
                'total_tokens': usage.total_tokens if usage else 0
            }
            
            return result_dict

    except json.JSONDecodeError:
        print(f"Ошибка: Модель {model} вернула некорректный JSON.")
        return None
    except httpx.RequestError as e:
        print(f"Сетевая ошибка прокси при запросе к OpenRouter: {e}")
        return None
    except Exception as e:
        print(f"Непредвиденная ошибка API OpenRouter (Модель: {model}): {e}")
        return None

def generate_text_by_cerebras_model(prompt: str, temperature: float = 0.8) -> dict:
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        with httpx.Client(proxy=proxy) as http_client:
            client = Cerebras(
                api_key=CEREBRAS_API_KEY, 
                http_client=http_client, 
                max_retries=1 
            )
            
            api_kwargs = {
                "model": cerebras_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4096,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "augmentation_schema",
                        "strict": True,
                        "schema": schema,
                    },
                },
                "top_p": 0.9
            }

            response = client.chat.completions.create(**api_kwargs)
            
            raw_content = response.choices[0].message.content
            result_dict = json.loads(raw_content)
            
            usage = response.usage
            result_dict['tokens'] = {
                'prompt_tokens': usage.prompt_tokens if usage else 0,
                'completion_tokens': usage.completion_tokens if usage else 0,
                'total_tokens': usage.total_tokens if usage else 0
            }
            
            return result_dict

    except json.JSONDecodeError:
        print("Ошибка: Модель вернула некорректный JSON.")
        return None
    except httpx.RequestError as e:
        print(f"Сетевая ошибка прокси: {e}")
        return None
    except Exception as e:
        print(f"Непредвиденная ошибка API Cerebras: {e}")
        time.sleep(10)
        return None
    

def collect_augmented_dataset(target_counts: dict, model_weights: dict, 
                              baseline_bleu: float = None, baseline_cos: float = None,
                              checkpoint_file='augmented_dataset.jsonl',
                              history_file='metrics_history.json'):

    current_counts = {label: 0 for label in target_counts.keys()}
    existing_texts = []
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                if not line.strip(): continue
                
                try:
                    data = json.loads(line)
                    
                    label = data.get('label')
                    if label in current_counts:
                        current_counts[label] += 1
                    
                    existing_texts.append(data.get('text'))
                
                except json.JSONDecodeError:
                    print(f"Предупреждение: пропущена битая строка {line_idx + 1} в файле {checkpoint_file}")
                    continue
    
    total_existing = len(existing_texts)

    total_calls = total_existing
    
    for label, target in target_counts.items():
        remaining = target - current_counts[label]
        
        if remaining <= 0:
            print(f"Класс {label} полностью собран ({current_counts[label]}/{target}). Пропускаем.")
            continue
            
        print(f"Текущий класс: {label}. Осталось сгенерировать: {remaining}")
        
        successful_generations = 0 
        
        while successful_generations < remaining:
            if label == 'listing':
                prompt = generate_listing_prompt() 
            elif label == 'single':
                prompt = generate_event_prompt() 
            else:
                prompt = generate_junk_prompt() 
                
            current_temp = round(random.uniform(0.3, 1.2), 2)
                
            model_name = random.choices(list(model_weights.keys()), weights=list(model_weights.values()), k=1)[0]
            print(f"[{label}] {successful_generations + 1}/{remaining} | Модель: {model_name} | Temp: {current_temp}...")            
            start_time = time.time()
            
            if "cerebras" in model_name.lower():
                result = generate_text_by_cerebras_model(prompt, temperature=current_temp)
            else:
                result = generate_text_by_openrouter(prompt, model=model_name, temperature=current_temp)
                
            latency_sec = round(time.time() - start_time, 2)
            
            if result is None or 'text' not in result:
                print("Ошибка соединения или парсинга. Пропускаем генерацию")
                continue
                
            generated_text = result['text']
            thought_process = result.get('thought_process', '') 
            tokens = result.get('tokens', {}) 
            
            record = {
                'label': label,
                'model': model_name,
                'temperature': current_temp,
                'latency_sec': latency_sec,
                'text_length_chars': len(generated_text),
                'tokens': tokens, 
                'thought_process': thought_process,
                'prompt': prompt, 
                'text': generated_text,
                'timestamp': str(datetime.datetime.now())
            }

            with open(checkpoint_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
                
            successful_generations += 1
            total_calls += 1

def return_targets_and_dataset(dataset_path: str = 'training_dataset.json'):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    targets = {
        'junk': 2333 - sum(1 for elem in dataset if elem['label'] == 'junk'),  
        'single': 2333 - sum(1 for elem in dataset if elem['label'] == 'single'), 
        'listing': 2334 - sum(1 for elem in dataset if elem['label'] == 'listing') 
    }

    return targets, dataset
    
if __name__ == '__main__':
    targets, dataset = return_targets_and_dataset()
    
    metrics = collect_augmented_dataset(targets, model_weights)
    plt.ioff() 
    plt.show()


