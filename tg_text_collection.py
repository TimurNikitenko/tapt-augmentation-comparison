import json
from dotenv import load_dotenv
import asyncio
import os
from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, FloodWaitError
from sources import TARGETS

load_dotenv()
SESSION_NAME = 'parser_session'
API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
OUTPUT_DIR = "parsed_data_telegram"

async def fetch_messages(client: TelegramClient, target: dict):
    name = target["name"]
    entity = target["entity"]
    topic_id = target.get("topic_id")
    limit = target.get("limit")
    
    print(f"[{name}] Начинаем сбор данных. Источник: {entity}" + 
          (f", Топик: {topic_id}" if topic_id else " (Весь чат/канал)"))
    
    messages_data = []
    
    try:
        # 1. Получаем объект сущности, чтобы узнать его параметры
        chat = await client.get_entity(entity)
        chat_username = getattr(chat, 'username', None)
        
        # Для приватных групп/каналов Telegram API возвращает ID с префиксом -100. 
        # В ссылках этот префикс не используется.
        chat_id_clean = str(chat.id).replace('-100', '')

        kwargs = {"limit": limit}
        if topic_id:
            kwargs["reply_to"] = topic_id

        async for message in client.iter_messages(entity, **kwargs):
            if message.text:
                # 2. Динамически формируем URL
                if chat_username:
                    # Публичный канал или супергруппа
                    msg_url = f"https://t.me/{chat_username}/{message.id}"
                else:
                    # Приватный канал или группа
                    msg_url = f"https://t.me/c/{chat_id_clean}/{message.id}"

                messages_data.append({
                    "id": message.id,
                    "url": msg_url,  
                    "date": message.date.isoformat(),
                    "text": message.text,
                })
                
        messages_data.reverse()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filename = os.path.join(OUTPUT_DIR, f"{name}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(messages_data, f, ensure_ascii=False, indent=4)
            
        print(f"[{name}] Успешно! Собрано сообщений: {len(messages_data)}. Файл: {filename}")
        
    except ChannelPrivateError:
        print(f"[{name}] Ошибка: Нет доступа к чату (закрытый канал или бан).")
    except ValueError as e:
        print(f"[{name}] Ошибка значения: Не удалось найти сущность {entity}. Проверьте ссылку.")
    except Exception as e:
        print(f"[{name}] Непредвиденная ошибка: {e}")

async def main():
    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    
    await client.start()
    print("Клиент авторизован.\n" + "="*40)
    
    for target in TARGETS:
        try:
            await fetch_messages(client, target)
            await asyncio.sleep(2) 
        except FloodWaitError as e:
            print(f"Сработал лимит Telegram. Ждем {e.seconds} секунд...")
            await asyncio.sleep(e.seconds)
            
    print("="*40 + "\nПарсинг завершен.")

if __name__ == '__main__':
    asyncio.run(main())