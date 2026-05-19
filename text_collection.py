import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def extract_raw_text(text_field):
    """
    Рекурсивно извлекает сырой текст из специфического формата Telegram экспорта.
    Telegram может хранить текст как строку или как список строк и словарей (для форматирования).
    """
    if isinstance(text_field, str):
        return text_field
    elif isinstance(text_field, list):
        combined_text = ""
        for item in text_field:
            if isinstance(item, str):
                combined_text += item
            elif isinstance(item, dict) and 'text' in item:
                combined_text += item['text']
        return combined_text
    return ""

def main():
    base_dir = Path('parsed_data_telegram')
    all_messages = []
    
    print(f"Ищем файлы .json в директории {base_dir}...")
    json_files = list(base_dir.rglob('*.json'))
    
    if not json_files:
        print("Файлы не найдены! Проверьте путь.")
        return

    for file_path in tqdm(json_files, desc="Парсинг каналов"):
        channel_name = file_path.stem 
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"\nОшибка чтения JSON в файле: {file_path}")
                continue
               
            for msg in data:
                raw_text = extract_raw_text(msg.get('text', ''))
                raw_text = raw_text.strip()
                url = msg.get('url', '')
                
                if raw_text:
                    all_messages.append({
                        'channel': channel_name,
                        'text': raw_text,
                        'text_length': len(raw_text.split()),
                        'url': url,
                    })

    df = pd.DataFrame(all_messages)
    
    if df.empty:
        print("Не найдено валидных сообщений.")
        return

    print("\n" + "="*50)
    print("Статистика")
    print("="*50)
    
    total_messages = len(df)
    total_channels = df['channel'].nunique()
    
    print(f"Всего обработано каналов: {total_channels}")
    print(f"Всего извлечено текстовых сообщений: {total_messages:,}".replace(',', ' '))
    print(f"Средняя длина сообщения: {df['text_length'].mean():.1f} слов")
    
    output_csv = 'extended_tg_dataset.csv'
    output_jsonl = 'extended_tg_dataset.jsonl'
    
    df[['channel', 'url', 'text']].to_csv(output_csv, index=False, encoding='utf-8')
    df[['channel', 'url', 'text']].to_json(output_jsonl, orient='records', lines=True, force_ascii=False)
    
    print(f"\nФайлы {output_csv} и {output_jsonl} успешно сохранены")

if __name__ == "__main__":
    main()