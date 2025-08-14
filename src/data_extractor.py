import types
from telethon_client import TelethonApiClient
from bot_client import AsyncTelegramBot
import os
import json
import logging
from typing import List, Dict, Optional
from telethon.tl.types import (
    Channel, User, Chat, Message, Document, Photo,
    DocumentAttributeFilename, ChannelParticipantsAdmins
)
from telethon.errors import ChannelPrivateError, ChatWriteForbiddenError
from telethon.tl.functions.channels import GetFullChannelRequest
from telethon.tl.functions.messages import GetFullChatRequest


DATA_DIR = "D:/projects/python/TelegramGraphNet/TeleNet/data"
os.makedirs(DATA_DIR, exist_ok=True)
# Настройка логирования
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Создаем необходимые директории
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "chats"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "messages"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "media"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "metadata"), exist_ok=True)



# -----------------------------
# Account data extraction
# -----------------------------


async def get_chat_full_info(client, chat_id):
    """Получение полной информации о чате с обработкой разных типов"""
    try:
        # Для каналов и супергрупп
        full = await client.client(GetFullChannelRequest(chat_id))
        return full.full_chat
    except:
        try:
            # Для обычных групп
            full = await client.client(GetFullChatRequest(chat_id))
            return full.full_chat
        except Exception as e:
            logger.error(f"Ошибка получения полной информации: {e}")
            return None


async def check_access(client, chat_id):
    """Проверяет, есть ли у нас доступ к чату"""
    try:
        await client.client.get_permissions(chat_id, await client.client.get_me())
        return True
    except (ChannelPrivateError, ChatWriteForbiddenError):
        return False
    except Exception as e:
        logger.warning(f"Ошибка проверки прав: {e}")
        return False


async def get_account_chats_account(client) -> List[dict]:
    """Получает информацию о всех чатах аккаунта"""
    logger.info("Начало сбора информации о чатах аккаунта")
    chats = []

    try:
        dialogs = await client.list_chats(limit=None)

        for dialog in dialogs:
            try:
                chat = dialog.entity
                chat_info = {
                    "id": chat.id,
                    "title": chat.title if hasattr(chat, 'title') else None,
                    "username": chat.username if hasattr(chat, 'username') else None,
                    "type": "private" if isinstance(chat, User) else
                    "channel" if isinstance(chat, Channel) and chat.broadcast else
                    "supergroup" if isinstance(chat, Channel) else
                    "group",
                    "participants_count": getattr(chat, 'participants_count', None),
                    "date": chat.date.isoformat() if hasattr(chat, 'date') else None,
                    "verified": getattr(chat, 'verified', False),
                    "scam": getattr(chat, 'scam', False),
                    "restricted": getattr(chat, 'restricted', False),
                    "access_hash": str(chat.access_hash) if hasattr(chat, 'access_hash') else None,
                }

                # Обработка фото профиля
                if chat.photo:
                    photo_info = {
                        "dc_id": chat.photo.dc_id,
                        "has_video": chat.photo.has_video,
                    }
                    if hasattr(chat.photo, 'photo_id'):
                        photo_info["photo_id"] = chat.photo.photo_id
                    chat_info["photo"] = photo_info
                else:
                    chat_info["photo"] = None

                # Для групп и каналов добавляем дополнительную информацию
                if not isinstance(chat, User):
                    try:
                        full_chat = await get_chat_full_info(client, chat.id)
                        if full_chat:
                            chat_info["full_info"] = {
                                "about": full_chat.about,
                                "online_count": full_chat.online_count,
                                "linked_chat_id": full_chat.linked_chat_id,
                                "slowmode_seconds": full_chat.slowmode_seconds
                            }
                    except Exception as e:
                        logger.warning(f"Ошибка получения информации о чате {chat.id}: {e}")

                chats.append(chat_info)

                # Логируем прогресс
                if len(chats) % 10 == 0:
                    logger.info(f"Обработано чатов: {len(chats)}")

            except Exception as e:
                logger.error(f"Ошибка обработки чата {dialog.id}: {e}")

        # Сохраняем результаты
        file_path = os.path.join(DATA_DIR, "chats", "account_chats.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chats, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Успешно собрана информация о {len(chats)} чатах")
        return chats

    except Exception as e:
        logger.error(f"Критическая ошибка при сборе чатов: {e}")
        return []


async def get_full_chat_history(client, chat_id: int) -> List[dict]:
    """Получение ВСЕХ сообщений чата без ограничений"""
    logger.info(f"Начало сбора ПОЛНОЙ истории чата {chat_id}")
    messages = []
    total = 0

    try:
        # Проверка доступа
        if not await check_access(client, chat_id):
            logger.warning(f"Нет доступа к чату {chat_id}, пропускаем сбор сообщений")
            return []

        # Получаем общее количество сообщений для прогресса
        entity = await client.client.get_entity(chat_id)
        if hasattr(entity, 'messages_count'):
            total = entity.messages_count
            logger.info(f"Всего сообщений в чате: {total}")

        # Собираем все сообщения
        async for message in client.client.iter_messages(entity):
            try:
                # Исправленная обработка реакций
                messages.append({
                    "id": message.id,
                    "date": message.date.isoformat(),
                    "text": message.text,
                    "sender_id": message.sender_id,
                    "views": message.views,
                    "forwards": message.forwards,
                    "replies": message.replies.replies if message.replies else 0,
                    "reactions": message.reactions.to_dict() if message.reactions else None
                })

                # Прогресс каждые 100 сообщений
                if len(messages) % 100 == 0:
                    logger.info(f"Собрано {len(messages)}/{total or '?'} сообщений")

            except Exception as e:
                logger.error(f"Ошибка обработки сообщения {message.id}: {e}")

        logger.info(f"Успешно собрано {len(messages)} сообщений")
        return messages

    except Exception as e:
        logger.error(f"Ошибка сбора истории чата {chat_id}: {e}")
        return []


async def get_message_data_account(client, chat_id: int) -> List[dict]:
    """Собирает ВСЕ сообщения из указанного чата"""
    logger.info(f"Начало сбора сообщений из чата {chat_id}")

    try:
        # Получаем полную историю
        history = await get_full_chat_history(client, chat_id)

        # Сохраняем результаты
        file_path = os.path.join(DATA_DIR, "messages", f"chat_{chat_id}_messages.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Собрано {len(history)} сообщений из чата {chat_id}")
        return history

    except Exception as e:
        logger.error(f"Ошибка сбора сообщений из чата {chat_id}: {e}")
        return []


async def get_media_data_account(client, chat_id: int) -> List[dict]:
    """Собирает и скачивает медиафайлы из указанного чата"""
    logger.info(f"Начало сбора медиа из чата {chat_id}")
    media_list = []

    try:
        # Проверка доступа
        if not await check_access(client, chat_id):
            logger.warning(f"Нет доступа к чату {chat_id}, пропускаем сбор медиа")
            return []

        # Создаем директорию для медиа
        media_dir = os.path.join(DATA_DIR, "media", f"chat_{chat_id}")
        os.makedirs(media_dir, exist_ok=True)

        # Получаем сообщения из чата
        messages = await get_full_chat_history(client, chat_id)

        for msg in messages:
            try:
                # Получаем полный объект сообщения
                full_message = await client.client.get_messages(chat_id, ids=msg["id"])

                if full_message and full_message.media:
                    media_info = {
                        "message_id": full_message.id,
                        "chat_id": chat_id,
                        "date": full_message.date.isoformat(),
                        "sender_id": full_message.sender_id,
                        "media_type": None,
                        "file_path": None,
                        "file_size": None,
                        "mime_type": None
                    }

                    # Обрабатываем разные типы медиа
                    if isinstance(full_message.media, Document):
                        doc = full_message.media.document
                        media_info["media_type"] = "document"
                        media_info["file_size"] = doc.size
                        media_info["mime_type"] = doc.mime_type

                        # Получаем имя файла
                        file_name = next(
                            (attr.file_name for attr in doc.attributes
                             if isinstance(attr, DocumentAttributeFilename)),
                            f"document_{doc.id}"
                        )

                        file_path = os.path.join(media_dir, file_name)
                        media_info["file_path"] = file_path

                        # Скачиваем файл
                        await client.client.download_media(
                            full_message.media,
                            file=file_path
                        )

                    elif isinstance(full_message.media, Photo):
                        photo = full_message.media.photo
                        media_info["media_type"] = "photo"

                        file_path = os.path.join(media_dir, f"photo_{photo.id}.jpg")
                        media_info["file_path"] = file_path

                        # Скачиваем фото
                        await client.client.download_media(
                            full_message.media,
                            file=file_path
                        )

                    # Добавляем информацию о медиа в список
                    media_list.append(media_info)

                    # Логируем прогресс
                    if len(media_list) % 10 == 0:
                        logger.info(f"Обработано медиафайлов: {len(media_list)}")

            except Exception as e:
                logger.error(f"Ошибка обработки медиа в сообщении {msg['id']}: {e}")

        # Сохраняем метаданные
        meta_path = os.path.join(DATA_DIR, "media", f"chat_{chat_id}_media.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(media_list, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Собрано {len(media_list)} медиафайлов из чата {chat_id}")
        return media_list

    except Exception as e:
        logger.error(f"Ошибка сбора медиа из чата {chat_id}: {e}")
        return []


async def get_metadata_account(client, chat_id: int) -> Dict:
    """Собирает метаданные о чате"""
    logger.info(f"Сбор метаданных для чата {chat_id}")
    metadata = {}

    try:
        # Проверка доступа
        if not await check_access(client, chat_id):
            logger.warning(f"Нет доступа к чату {chat_id}, пропускаем сбор метаданных")
            return {}

        # Основная информация о чате
        entity = await client.client.get_entity(chat_id)
        metadata["basic_info"] = {
            "id": entity.id,
            "title": entity.title if hasattr(entity, 'title') else None,
            "username": entity.username if hasattr(entity, 'username') else None,
            "date": entity.date.isoformat() if hasattr(entity, 'date') else None,
            "verified": getattr(entity, 'verified', False),
            "scam": getattr(entity, 'scam', False),
            "participants_count": getattr(entity, 'participants_count', None),
            "access_hash": str(entity.access_hash) if hasattr(entity, 'access_hash') else None
        }

        # Полная информация о чате
        try:
            full_chat = await get_chat_full_info(client, chat_id)
            if full_chat:
                metadata["full_info"] = {
                    "about": full_chat.about,
                    "online_count": full_chat.online_count,
                    "linked_chat_id": full_chat.linked_chat_id,
                    "slowmode_seconds": full_chat.slowmode_seconds
                }
        except Exception as e:
            logger.warning(f"Не удалось получить полную информацию о чате: {e}")

        # Участники чата (только если есть права)
        participants = []
        try:
            async for user in client.client.iter_participants(chat_id):
                try:
                    last_online = None
                    if user.status:
                        if hasattr(user.status, 'was_online'):
                            last_online = user.status.was_online.isoformat()

                    participants.append({
                        "id": user.id,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "username": user.username,
                        "is_bot": user.bot,
                        "is_verified": user.verified,
                        "is_scam": user.scam,
                        "is_premium": user.premium,
                        "last_online": last_online,
                        "status": str(user.status),
                        "lang_code": user.lang_code
                    })

                    # Логируем прогресс
                    if len(participants) % 50 == 0:
                        logger.info(f"Собрано участников: {len(participants)}")

                except Exception as e:
                    logger.error(f"Ошибка обработки участника {user.id}: {e}")

            metadata["participants"] = participants
        except Exception as e:
            logger.warning(f"Не удалось получить участников чата: {e}")
            metadata["participants"] = []

        # Администраторы чата
        admins = []
        try:
            async for admin in client.client.iter_participants(
                    chat_id,
                    filter=ChannelParticipantsAdmins
            ):
                try:
                    admin_info = {
                        "id": admin.id,
                        "first_name": admin.first_name,
                        "last_name": admin.last_name,
                        "username": admin.username
                    }

                    # Для администраторов
                    if hasattr(admin, 'admin_rights') and admin.admin_rights:
                        admin_info["admin_rights"] = {
                            "change_info": admin.admin_rights.change_info,
                            "post_messages": admin.admin_rights.post_messages,
                            "edit_messages": admin.admin_rights.edit_messages,
                            "delete_messages": admin.admin_rights.delete_messages,
                            "ban_users": admin.admin_rights.ban_users,
                            "invite_users": admin.admin_rights.invite_users,
                            "pin_messages": admin.admin_rights.pin_messages,
                            "add_admins": admin.admin_rights.add_admins
                        }

                    # Для создателей
                    if hasattr(admin, 'title'):
                        admin_info["title"] = admin.title

                    admins.append(admin_info)
                except Exception as e:
                    logger.error(f"Ошибка обработки администратора {admin.id}: {e}")

            metadata["admins"] = admins
        except Exception as e:
            logger.warning(f"Не удалось получить администраторов чата: {e}")
            metadata["admins"] = []

        # Сохраняем результаты
        file_path = os.path.join(DATA_DIR, "metadata", f"chat_{chat_id}_metadata.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Собраны метаданные для чата {chat_id}")
        return metadata

    except Exception as e:
        logger.error(f"Ошибка сбора метаданных для чата {chat_id}: {e}")
        return {}

# -----------------------------
# Bot data extraction
# -----------------------------


async def get_account_chats_bot(bot: AsyncTelegramBot, chat_ids: List[int]) -> List[dict]:
    chats = []
    for cid in chat_ids:
        try:
            # Получаем основную информацию о чате
            chat = await bot.bot.get_chat(cid)

            # Инициализируем переменные
            admins = []
            members_count = 0
            permissions = None

            try:
                # Для личных чатов нет администраторов, пропускаем
                if chat.type != "private":
                    admins = await bot.bot.get_chat_administrators(cid)
                    members_count = await bot.bot.get_chat_member_count(cid)
            except Exception as e:
                print(f"Ошибка при получении доп. информации для чата {cid}: {e}")

            # Обработка разрешений без использования to_dict()
            if hasattr(chat, 'permissions') and chat.permissions is not None:
                permissions = {
                    'can_send_messages': chat.permissions.can_send_messages,
                    'can_send_media_messages': chat.permissions.can_send_media_messages,
                    'can_send_polls': chat.permissions.can_send_polls,
                    'can_send_other_messages': chat.permissions.can_send_other_messages,
                    'can_add_web_page_previews': chat.permissions.can_add_web_page_previews,
                    'can_change_info': chat.permissions.can_change_info,
                    'can_invite_users': chat.permissions.can_invite_users,
                    'can_pin_messages': chat.permissions.can_pin_messages,
                }

            # Формируем информацию о чате
            chat_info = {
                "chat_id": cid,
                "title": chat.title,
                "type": chat.type,
                "description": getattr(chat, 'description', None),
                "username": getattr(chat, 'username', None),
                "members_count": members_count,
                "admins": [{
                    "id": admin.user.id,
                    "username": admin.user.username,
                    "first_name": admin.user.first_name,
                    "last_name": admin.user.last_name
                } for admin in admins],
                "permissions": permissions,
                "created_at": chat.date.isoformat() if hasattr(chat, 'date') else None
            }
            chats.append(chat_info)
        except Exception as e:
            print(f"Ошибка получения информации о чате {cid}: {e}")

    file_path = os.path.join(DATA_DIR, "bot_chats.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2, default=str)

    return chats

async def get_message_data_bot(bot: AsyncTelegramBot, chat_ids: List[int], limit: Optional[int] = None) -> List[dict]:
    all_messages = []

    for chat_id in chat_ids:
        try:
            print(f"Сбор сообщений из чата {chat_id}...")
            chat_messages = []
            count = 0

            # Получаем историю сообщений для каждого чата
            async for message in bot.get_history(chat_id):
                try:
                    # Извлекаем информацию об отправителе
                    user = message.from_user
                    sender = {
                        "id": user.id,
                        "username": user.username,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "is_bot": user.is_bot
                    } if user else None

                    # Извлекаем информацию о форварде
                    forward = None
                    if message.forward_from:
                        forward = {
                            "from_id": message.forward_from.id,
                            "from_name": f"{message.forward_from.first_name} {message.forward_from.last_name or ''}"
                        }
                    elif message.forward_from_chat:
                        forward = {
                            "from_chat_id": message.forward_from_chat.id,
                            "from_chat_name": message.forward_from_chat.title
                        }

                    # Формируем сообщение
                    msg_data = {
                        "chat_id": chat_id,
                        "message_id": message.message_id,
                        "date": message.date.isoformat(),
                        "sender": sender,
                        "text": message.text or message.caption or "",
                        "entities": [e.to_dict() for e in (message.entities or [])],
                        "caption_entities": [e.to_dict() for e in (message.caption_entities or [])],
                        "forward": forward,
                        "reply_to": message.reply_to_message.message_id if message.reply_to_message else None,
                        "media_type": message.content_type,
                        "has_media": bool(message.photo or message.document or message.video),
                        "views": getattr(message, 'views', None),
                        "forwards": getattr(message, 'forward_date', None),
                    }

                    chat_messages.append(msg_data)
                    count += 1

                    if limit and count >= limit:
                        print(f"Достигнут лимит в {limit} сообщений для чата {chat_id}")
                        break

                except Exception as e:
                    print(f"Ошибка обработки сообщения {message.message_id} в чате {chat_id}: {e}")

            print(f"Собрано {len(chat_messages)} сообщений из чата {chat_id}")
            all_messages.extend(chat_messages)

        except Exception as e:
            print(f"Ошибка доступа к чату {chat_id}: {e}")
            # Для каналов и некоторых групп могут потребоваться особые права
            if "not found" in str(e).lower():
                print("Убедитесь, что бот добавлен в чат и имеет права администратора")
            elif "Forbidden" in str(e):
                print("Бот не имеет доступа к этому чату")

    file_path = os.path.join(DATA_DIR, "bot_messages.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=2)

    return all_messages

async def get_media_data_bot(bot: AsyncTelegramBot, limit: Optional[int] = None) -> List[dict]:
    media_list = []
    count = 0
    tasks = []
    semaphore = asyncio.Semaphore(5)  # Ограничение параллельных загрузок

    async def process_media(message: types.Message):
        nonlocal count
        async with semaphore:
            try:
                if message.photo:
                    file_id = message.photo[-1].file_id
                    media_type = "photo"
                elif message.document:
                    file_id = message.document.file_id
                    media_type = "document"
                elif message.video:
                    file_id = message.video.file_id
                    media_type = "video"
                elif message.audio:
                    file_id = message.audio.file_id
                    media_type = "audio"
                else:
                    return

                # Создаем директорию для медиа
                chat_dir = os.path.join(DATA_DIR, "media", str(message.chat.id))
                os.makedirs(chat_dir, exist_ok=True)

                # Получаем информацию о файле
                file = await bot.bot.get_file(file_id)
                extension = file.file_path.split('.')[-1] if file.file_path else 'bin'
                filename = f"{message.message_id}_{file_id[:8]}.{extension}"
                file_path = os.path.join(chat_dir, filename)

                # Скачиваем файл
                await bot.bot.download_file(file.file_path, destination=file_path)

                media_list.append({
                    "chat_id": message.chat.id,
                    "message_id": message.message_id,
                    "file_id": file_id,
                    "file_path": file_path,
                    "file_size": file.file_size,
                    "media_type": media_type,
                    "date": message.date.isoformat()
                })

                count += 1
            except Exception as e:
                print(f"Ошибка загрузки медиа: {e}")

    async for upd in bot.get_updates(timeout=5):
        if upd.message:
            if limit and count >= limit:
                break
            tasks.append(asyncio.create_task(process_media(upd.message)))

    await asyncio.gather(*tasks)

    file_path = os.path.join(DATA_DIR, "bot_media.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(media_list, f, ensure_ascii=False, indent=2)

    return media_list

async def get_all_data_bot(bot: AsyncTelegramBot, chat_ids: List[int], limit: Optional[int] = None) -> None:
    # Параллельный сбор данных
    chats_task = asyncio.create_task(get_account_chats_bot(bot, chat_ids))
    messages_task = asyncio.create_task(get_message_data_bot(bot, limit))
    media_task = asyncio.create_task(get_media_data_bot(bot, limit))

    # Ожидаем завершения всех задач
    await asyncio.gather(chats_task, messages_task, media_task)

    result = {
        "chats": await chats_task,
        "messages": await messages_task,
        "media": await media_task
    }

    file_path = os.path.join(DATA_DIR, "all_data_bot.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def run_bot_extraction(chat_ids: List[int], limit=None):
    async def _inner():
        bot = AsyncTelegramBot()
        try:
            print("Начало сбора данных...")
            await get_all_data_bot(bot, chat_ids, limit)
            print("Сбор данных завершен успешно")
        except Exception as e:
            print(f"Ошибка при сборе данных: {e}")
        finally:
            await bot.close()

    asyncio.run(_inner())


# -----------------------------
# Helpfull functions
# -----------------------------


async def main():


    # chat_id = 2812723486
    # await get_message_data_account(client, chat_id)
    # await get_account_chats_account(client)
    async with TelethonApiClient() as client:
        # Получаем список чатов
        chats = await get_account_chats_account(client)

        # Для каждого чата собираем полные данные
        for chat in chats:
            chat_id = chat["id"]

            # Собираем все сообщения
            await get_message_data_account(client, chat_id)


            # Собираем метаданные
            await get_metadata_account(client, chat_id)

            logger.info(f"Завершен сбор данных для чата {chat_id}")



if __name__ == "__main__":
    import asyncio

    asyncio.run(main())