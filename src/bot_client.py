import asyncio
from aiogram.types import Message
from typing import AsyncGenerator, Optional, List

from aiogram.types import Message as AiogramMessage
from aiogram import Bot, types
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from dotenv import load_dotenv
import os

class AsyncTelegramBot:
    def __init__(self: str):
        load_dotenv()
        self.token = os.getenv("BOT_TOKEN")
        self.bot = Bot(token=self.token)
        self.updates_offset = 0  # для хранения последнего обработанного апдейта
        self.last_message_ids = {}

    async def connect(self):
        # Никаких дополнительных действий не требуется, Bot умеет работать сразу
        # (в aiogram нет явного метода .connect(), можно сразу вызывать send/receive).
        pass

    async def send_message(self, chat_id: int, text: str):
        """Отправить текстовое сообщение в чат."""
        try:
            await self.bot.send_message(chat_id, text)
        except TelegramBadRequest as e:
            print(f"Ошибка при отправке: {e}")

    async def forward_message(self, chat_id: int, from_chat_id: int, message_id: int):
        """Переслать сообщение из одного чата в другой."""
        await self.bot.forward_message(chat_id=chat_id, from_chat_id=from_chat_id, message_id=message_id)

    async def download_media(self, file_id: str, path: str):
        """Скачать файл по file_id в указанный путь."""
        # Сначала получаем объект File
        file = await self.bot.get_file(file_id)
        await self.bot.download_file(file.file_path, path)

    async def get_chat_info(self, chat_id: int):
        """Получить информацию о чате (название, участников и т.д.)."""
        chat = await self.bot.get_chat(chat_id)
        admins = await self.bot.get_chat_administrators(chat_id)
        count = await self.bot.get_chat_member_count(chat_id)
        return {
            'title': chat.title,
            'type': chat.type,
            'members_count': count,
            'admins': [admin.user.id for admin in admins]
        }

    async def get_updates(self, timeout: int = 10):
        """Получить новые апдейты (long polling)."""
        updates = await self.bot.get_updates(offset=self.updates_offset, timeout=timeout, allowed_updates=['message','callback_query','message_reaction','message_reaction_count'])
        for upd in updates:
            self.updates_offset = max(self.updates_offset, upd.update_id + 1)
            yield upd

    async def handle_updates(self):
        """Основной цикл обработки апдейтов."""
        async for update in self.get_updates():
            if update.message:
                msg = update.message
                print(f"Новое сообщение от {msg.from_user.id}: {msg.text}")
                # Пример: отправить ответ
                await self.send_message(msg.chat.id, f"Вы написали: {msg.text}")
            if update.message_reaction:
                reaction = update.message_reaction
                print(f"Реакция в чате {reaction.chat.id}: {reaction.reaction}")

    async def close(self):
        """Закрыть соединение с Bot API."""
        await self.bot.session.close()

    async def list_active_chats(self):
        """
        Получает список всех чатов, где присутствует бот, с названиями чатов
        Возвращает словарь {chat_id: chat_title}
        """
        active_chats = {}

        # Получаем все доступные обновления (макс. 100 последних)
        try:
            updates = await self.bot.get_updates(offset=0, timeout=1)
        except Exception as e:
            print(f"Ошибка при получении обновлений: {e}")
            return active_chats

        # Собираем уникальные ID чатов из всех типов обновлений
        for update in updates:
            chat = None

            if update.message:
                chat = update.message.chat
            elif update.edited_message:
                chat = update.edited_message.chat
            elif update.channel_post:
                chat = update.channel_post.chat
            elif update.callback_query and update.callback_query.message:
                chat = update.callback_query.message.chat
            elif update.message_reaction:
                chat = update.message_reaction.chat

            if chat:
                active_chats[chat.id] = chat.title

        # Дополнительно проверяем чаты из информации о боте
        try:
            me = await self.bot.get_me()
            user_chats = await self.bot.get_chat(me.id)
            if hasattr(user_chats, 'private_chats'):
                for chat in user_chats.private_chats.values():
                    active_chats[chat.id] = chat.title
        except Exception as e:
            print(f"Ошибка при получении личных чатов: {e}")

        # Выводим результат
        print("\n" + "=" * 50)
        print(f"Бот присутствует в {len(active_chats)} чатах:")
        for chat_id, title in active_chats.items():
            print(f"ID: {chat_id} | Название: {title}")
        print("=" * 50)

        return active_chats




async def main():
    # Инициализация бота
    bot = AsyncTelegramBot()

    # Получение списка чатов
    # chats = await bot.list_active_chats()

    # Дополнительные действия с полученными данными
    # print(f"Всего чатов: {len(chats)}")
    await bot.send_message(chat_id=-1002812723486,text="@Timursedov, сам соси")

    # Закрытие соединения
    await bot.close()


# Запуск асинхронной функции
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())