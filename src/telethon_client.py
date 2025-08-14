import os, logging
import time
import asyncio
from dotenv import load_dotenv
from telethon import TelegramClient, errors
from telethon.errors import FloodWaitError, ChannelPrivateError, ChatWriteForbiddenError


class TelethonApiClient:
    """
    Улучшенный Telethon-клиент с обработкой FloodWait и ошибок
    """

    def __init__(self, max_retries=3, retry_delay=5):
        # Загружаем переменные окружения из .env
        load_dotenv()
        self.api_id = int(os.getenv("API_ID", "0"))
        self.api_hash = os.getenv("API_HASH", "")
        self.session_name = os.getenv("SESSION_NAME", "anon")
        self.phone = os.getenv("TELEGRAM_PHONE", "")
        self.system_version = '4.16.30-vxCUSTOM'
        self.max_retries = max_retries  # Максимальное количество попыток
        self.retry_delay = retry_delay  # Базовая задержка между попытками (секунды)

        if not self.api_id or not self.api_hash:
            raise ValueError("Не заданы API_ID или API_HASH в .env")

        # Настраиваем логирование
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Инициализируем Telethon-клиент
        self.client = TelegramClient(self.session_name, self.api_id, self.api_hash, system_version=self.system_version)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def _handle_flood_wait(self, error):
        """Обработка ошибки FloodWait с автоматической паузой"""
        wait_time = error.seconds + 5  # Добавляем запас в 5 секунд
        self.logger.warning(f"FloodWait: ожидание {wait_time} секунд...")
        await asyncio.sleep(wait_time)
        self.logger.info("Продолжение работы после FloodWait")

    async def _safe_request(self, coro, *args, **kwargs):
        """
        Безопасное выполнение запроса с обработкой ошибок и повторными попытками
        """
        for attempt in range(self.max_retries):
            try:
                return await coro(*args, **kwargs)
            except FloodWaitError as e:
                await self._handle_flood_wait(e)
            except (ChannelPrivateError, ChatWriteForbiddenError) as e:
                self.logger.error(f"Нет доступа: {e}")
                raise
            except errors.RPCError as e:
                self.logger.error(f"Ошибка Telegram API: {e}")
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (attempt + 1)
                    self.logger.warning(f"Повторная попытка через {delay} сек...")
                    await asyncio.sleep(delay)
                else:
                    raise
            except Exception as e:
                self.logger.exception(f"Неизвестная ошибка: {e}")
                raise
        return None

    async def connect(self):
        """Подключаемся к Telegram с обработкой ошибок"""
        self.logger.info("Подключение к Telegram...")
        try:
            await self._safe_request(self.client.start, phone=self.phone)
        except OSError as e:
            self.logger.error(f"Не удалось подключиться: {e}")
            raise

        # Если не авторизованы, выполняем вход по коду
        if not await self.client.is_user_authorized():
            if not self.phone:
                raise ValueError("Не указан телефон в TELEGRAM_PHONE для входа.")

            self.logger.info("Не авторизован. Запрос кода подтверждения...")
            await self._safe_request(self.client.send_code_request, self.phone)

            code = input("Введите код из Telegram: ")
            try:
                await self._safe_request(self.client.sign_in, self.phone, code)
            except errors.SessionPasswordNeededError:
                # Если включена 2FA
                pwd = input("Введите пароль (2FA): ")
                await self._safe_request(self.client.sign_in, password=pwd)

        self.logger.info("Успешно подключено и авторизовано.")

    async def disconnect(self):
        """Отключаемся от Telegram."""
        self.logger.info("Отключение от Telegram...")
        await self.client.disconnect()
        self.logger.info("Отключено.")

    async def send_message(self, target, text):
        """
        Отправка сообщения в чат/канал/ЛС.
        Автоматически определяет, можно ли использовать send_as.
        """
        try:
            # Получаем entity
            entity = await self._safe_request(self.client.get_entity, target)
            self.logger.info(f"Отправка сообщения в {target}")

            # Получаем данные о себе
            me = await self.client.get_me()

            # Проверяем, можно ли использовать send_as
            send_as_allowed = False
            if hasattr(entity, 'creator') and entity.creator:
                # Если это канал, в котором вы являетесь создателем
                send_as_allowed = True
            elif hasattr(entity, 'broadcast') and entity.broadcast:
                # Если это публичный канал, отправка от имени не разрешена
                send_as_allowed = False
            # Для личного чата send_as запрещён
            # Для обычной группы тоже запрещено

            if send_as_allowed:
                result = await self._safe_request(
                    self.client.send_message,
                    entity,
                    text,
                    send_as=me
                )
            else:
                result = await self._safe_request(
                    self.client.send_message,
                    entity,
                    text
                )

            self.logger.info(f"Сообщение успешно отправлено (ID: {result.id})")
            return result

        except Exception as e:
            self.logger.error(f"Ошибка отправки сообщения: {e}")
            raise

    async def get_messages(self, target, limit=10):
        """
        Получение сообщений с обработкой ошибок
        """
        try:
            entity = await self._safe_request(self.client.get_entity, target)
            self.logger.info(f"Получение последних {limit} сообщений из {target}")
            return await self._safe_request(self.client.get_messages, entity, limit=limit)
        except Exception as e:
            self.logger.error(f"Ошибка получения сообщений: {e}")
            raise

    async def forward_messages(self, target, messages, from_target=None):
        """
        Пересылка сообщений с обработкой ошибок
        """
        try:
            dest = await self._safe_request(self.client.get_entity, target)
            src = await self._safe_request(self.client.get_entity, from_target) if from_target else None

            self.logger.info(f"Пересылка {len(messages)} сообщений в {target}")
            return await self._safe_request(
                self.client.forward_messages,
                dest,
                messages,
                from_peer=src
            )
        except Exception as e:
            self.logger.error(f"Ошибка пересылки сообщений: {e}")
            raise

    async def download_media(self, target, message_id, filename=None):
        """
        Скачивание медиа с обработкой ошибок
        """
        try:
            entity = await self._safe_request(self.client.get_entity, target)
            self.logger.info(f"Скачивание медиа из сообщения {message_id} чата {target}")

            msg = await self._safe_request(self.client.get_messages, entity, ids=message_id)
            if not msg:
                self.logger.warning("Сообщение с указанным ID не найдено.")
                return None

            path = await self._safe_request(self.client.download_media, msg, file=filename)
            self.logger.info(f"Медиа успешно сохранено в {path}")
            return path
        except Exception as e:
            self.logger.error(f"Ошибка скачивания медиа: {e}")
            raise

    async def list_chats(self, limit=100):
        """
        Получение списка чатов с обработкой ошибок
        """
        self.logger.info("Получение списка чатов (диалогов)")
        try:
            return await self._safe_request(self.client.get_dialogs, limit)
        except Exception as e:
            self.logger.error(f"Ошибка получения списка чатов: {e}")
            raise

    async def get_full_chat_history(self, chat_id, limit=None):
        """Получение полной истории чата с обработкой ошибок"""
        try:
            entity = await self._safe_request(self.client.get_entity, chat_id)
            self.logger.info(f"Начало сбора истории чата {chat_id} (лимит: {limit or 'без ограничений'})")

            messages = []
            async for message in self.client.iter_messages(entity, limit=limit):
                try:
                    messages.append({
                        "id": message.id,
                        "date": message.date.isoformat(),
                        "text": message.text,
                        "sender_id": message.sender_id,
                        "views": message.views,
                        "forwards": message.forwards,
                        "replies": message.replies.replies if message.replies else 0,
                        "reactions": [r.to_dict() for r in message.reactions] if message.reactions else []
                    })

                    # Логируем прогресс каждые 100 сообщений
                    if len(messages) % 100 == 0:
                        self.logger.info(f"Собрано {len(messages)} сообщений...")
                except Exception as e:
                    self.logger.error(f"Ошибка обработки сообщения {message.id}: {e}")

            self.logger.info(f"Успешно собрано {len(messages)} сообщений")
            return messages
        except Exception as e:
            self.logger.error(f"Ошибка сбора истории чата: {e}")
            raise


async def main():
    async with TelethonApiClient(max_retries=5, retry_delay=10) as client:
        chat = os.getenv("TELEGRAM_CHAT")

        try:
            # Отправим сообщение
            await client.send_message(target=chat, text='Проверка корректности работы API управления аккаунтом')

            # Получим последние 5 сообщений
            # msgs = await client.get_messages(chat, limit=5)
            # for m in msgs:
            #     print(m.sender_id, m.text)

            # Выведем список диалогов
            # dialogs = await client.list_chats(limit=10)
            # for d in dialogs[:5]:
            #     print(d.title or d.name)

            # Соберем историю небольшого чата (первые 50 сообщений)
            # history = await client.get_full_chat_history(chat, limit=50)
            # print(f"Собрано {len(history)} сообщений истории")

        except Exception as e:
            logging.error(f"Критическая ошибка в main: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())