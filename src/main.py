from telethon_client import TelethonApiClient
from bot_client import AsyncTelegramBot
import os
from telethon.sync import TelegramClient
from telethon.sessions import StringSession
from dotenv import load_dotenv

import asyncio

async def main():
    async with TelethonApiClient() as client:
        chat = os.getenv("TELEGRAM_CHAT")

        # Отправим сообщение
        await client.send_message(chat, 'Сообщение отправлено через API. Успех!')
        # Получим последние 5 сообщений
        msgs = await client.get_messages(chat, limit=5)
        for m in msgs:
            print(m.sender_id, m.text)


        # Выведем список диалогов
        dialogs = await client.list_chats()
        for d in dialogs[:5]:
            print(d.title or d.name)

# Пример использования класса
async def bot_used(token):

    bot = AsyncTelegramBot()
    await bot.send_message(chat_id=123456789, text="Бот запущен!")
    # Запустить обработку входящих сообщений (в бесконечном цикле)
    await bot.handle_updates()







