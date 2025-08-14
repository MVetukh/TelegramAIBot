# prepare_dataset.py
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------- CONFIG: укажи свои пути и параметры ----------
INPUT_JSON = r"D:/projects/python/TelegramGraphNet/TeleNet/data/messages/chat_759575591_messages.json"   # <- путь к исходному JSON с сообщениями
OUT_JSONL = r"D:/projects/python/TelegramGraphNet/TeleNet/dataset/chat_759575591_messages_dataset.jsonl"         # <- куда сохранить pairs в формате jsonl
WINDOW = 5                              # <- сколько предыдущих сообщений брать в prompt
TARGET = None                           # <- id пользователя (int) или None для всех
# ---------------------------------------------------------


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iso_to_dt(s: Optional[str]) -> datetime:
    if not s:
        return datetime.min
    try:
        # устойчивый парсер ISO-формата (учитываем возможный Z)
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        try:
            # fallback: попытка parse с более простым форматом
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
        except Exception:
            return datetime.min

def extract_text_field(msg: Dict[str, Any]) -> str:
    """
    Безопасно извлекает текст из структуры сообщения экспортированного Telethon-API.
    Телеграм-экспорт иногда содержит:
      - "text": строка или None
      - "message": строка
      - "text" как список сегментов (entity-список) -> объединяем
    Возвращает пустую строку, если текста нет.
    """
    if not isinstance(msg, dict):
        return ""

    # Попытка стандартного ключа 'text'
    text = msg.get("text", None)
    if isinstance(text, str):
        return text.strip()

    # Иногда содержится 'message'
    if isinstance(msg.get("message", None), str):
        return msg["message"].strip()

    # Если 'text' — список сегментов (частая структура), соберём всё строковое содержимое
    if isinstance(text, list):
        parts = []
        for seg in text:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict) and "text" in seg and isinstance(seg["text"], str):
                parts.append(seg["text"])
        joined = "".join(parts).strip()
        return joined

    # Иногда поле может быть None или другого типа — вернём пустую строку
    return ""

def prepare_pairs(messages: List[Dict[str, Any]], window_size: int = 3, target_sender_id: Optional[int] = None):
    # сортируем по дате (возрастание)
    messages = sorted(messages, key=lambda m: iso_to_dt(m.get("date", "")))
    pairs = []
    skipped_no_text = 0
    skipped_incomplete_window = 0

    for i in range(window_size, len(messages)):
        prev = messages[i-window_size:i]  # list of dicts
        cur = messages[i]

        # Получаем completion текст
        completion = extract_text_field(cur)
        if not completion:
            skipped_no_text += 1
            continue

        # если target задан, берем только случаи где cur.sender_id == target
        if target_sender_id is not None and cur.get("sender_id") != target_sender_id:
            continue

        # Формируем prompt из prev; пропускаем, если в одном из prev нет текста
        prompt_lines = []
        missing = False
        for msg in prev:
            txt = extract_text_field(msg)
            if not txt:
                missing = True
                break
            # метки ролей: если задан target, помечаем его как Target, иначе используем generic User/Other
            if target_sender_id is not None:
                role = "Target" if msg.get("sender_id") == target_sender_id else "User"
            else:
                role = "User" if msg.get("sender_id") != cur.get("sender_id") else "Other"
            prompt_lines.append(f"{role}: {txt}")
        if missing:
            skipped_incomplete_window += 1
            continue

        prompt = "\n".join(prompt_lines).strip()
        if prompt and completion:
            pairs.append({"prompt": prompt, "completion": completion})

    return pairs, skipped_no_text, skipped_incomplete_window

def save_jsonl(pairs: List[Dict[str, str]], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Подготовка путей
    input_path = Path(INPUT_JSON)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path.resolve()}")

    out_path = Path(OUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading messages from {input_path.resolve()} ...")
    msgs = load_json(str(input_path))
    print(f"Loaded {len(msgs)} messages")

    pairs, skipped_no_text, skipped_incomplete_window = prepare_pairs(msgs, window_size=WINDOW, target_sender_id=TARGET)
    save_jsonl(pairs, str(out_path))

    print(f"Saved {len(pairs)} pairs to {out_path.resolve()}")
    print(f"Skipped messages with no completion text: {skipped_no_text}")
    print(f"Skipped windows with missing context messages: {skipped_incomplete_window}")
