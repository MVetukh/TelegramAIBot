"""
Telegram LLM Bot (переработанный код с соблюдением модульности и классов)
"""
import os
import re
import asyncio
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

# Импорт улучшенного Telethon-клиента
from telethon_client import TelethonApiClient

# Попытка импортировать трансформеры и PyTorch
try:
    import torch  # noqa: F401
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Для HF Inference API (альтернативный бэкенд)
try:
    import aiohttp
except ImportError:
    aiohttp = None

# Загрузка переменных окружения
load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("telegram-llm-bot")

class LocalLLM:
    """
    Обёртка для локальной генерации текста с помощью Transformers (PyTorch).
    Поддерживает многопоточность через ThreadPoolExecutor.
    """
    def __init__(self, model_name: str = "gpt2", device: int = -1, max_workers: int = 1):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers / torch не установлены или недоступны в этом окружении.")
        self.model_name = model_name
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._load_pipeline()

    def _load_pipeline(self):
        logger.info(f"Загружаем модель {self.model_name} (это может занять время)...")
        # Если указан путь к директории с LoRA-адаптациями, пробуем загрузить её.
        if os.path.isdir(self.model_name) and os.path.isfile(os.path.join(self.model_name, "adapter_config.json")):
            try:
                from peft import PeftModel, PeftConfig
                peft_config = PeftConfig.from_pretrained(self.model_name)
                base_model_name = peft_config.base_model_name_or_path
                logger.info(f"LoRA конфигурация найдена, базовая модель: {base_model_name}")
                # Загружаем базовую модель и применяем LoRA
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
                self.model = PeftModel.from_pretrained(base_model, self.model_name)
            except ImportError:
                logger.error("Для использования LoRA необходимо установить библиотеку `peft`. Переходим к обычной загрузке модели.")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        else:
            # Обычная загрузка модели
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        device_arg = self.device if isinstance(self.device, int) else -1
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_arg
        )
        logger.info(f"Устройство для генерации установлено: {'CPU' if device_arg == -1 else 'CUDA:'+str(device_arg)}")

    async def generate(self, prompt: str,
                       max_new_tokens: int = 150,
                       do_sample: bool = True,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       repetition_penalty: float = 1.2,
                       no_repeat_ngram_size: int = 3) -> str:
        loop = asyncio.get_running_loop()
        def blocking():
            out = self.pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            return out[0]["generated_text"] if out else ""
        return await loop.run_in_executor(self.executor, blocking)


class HfApiLLM:
    """
    Обёртка для Hugging Face Inference API (асинхронно через aiohttp).
    """
    def __init__(self, model: str = "gpt2", hf_token: str = None):
        if aiohttp is None:
            raise RuntimeError("aiohttp требуется для HF inference backend. Установите `aiohttp`.")
        self.model = model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise RuntimeError("HF_TOKEN не задан (переменная окружения или параметр).")

    async def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "return_full_text": False
            }
        }
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                txt = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HF inference error {resp.status}: {txt}")
                data = await resp.json()
                if isinstance(data, list) and data and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                if isinstance(data, dict) and "generated_text" in data:
                    return data["generated_text"]
                return str(data)


def create_llm_from_env():
    """
    Создаёт экземпляр LLM в соответствии с переменными окружения.
    Локальный бэкенд приоритетен, если доступен, иначе HF API.
    """
    backend = os.getenv("LLM_BACKEND", "auto").lower()  # auto | local | hf
    model_name = os.getenv("LLM_MODEL", "gpt2")
    hf_token = os.getenv("HF_TOKEN")
    device = int(os.getenv("LLM_DEVICE", "-1"))
    max_workers = int(os.getenv("LLM_MAX_WORKERS", "1"))

    use_local = (backend == "local") or (backend == "auto" and TRANSFORMERS_AVAILABLE)
    if use_local:
        logger.info("Используем локальный Transformers backend")
        return LocalLLM(model_name=model_name, device=device, max_workers=max_workers)
    else:
        logger.info("Используем Hugging Face Inference API backend")
        return HfApiLLM(model=model_name, hf_token=hf_token)


class TelegramLLMAgent:
    """
    Агент, объединяющий Telethon и LLM для обработки входящих сообщений.
    """
    def __init__(self, tele_client: TelethonApiClient, llm, concurrency: int = 1, max_history: int = 12):
        self.tele = tele_client
        self.llm = llm
        self.semaphore = asyncio.Semaphore(concurrency)
        self.histories = defaultdict(lambda: deque(maxlen=max_history))
        # Системная инструкция (роль ассистента)
        self.system_instruction = (
            "Ты дружелюбный и общительный молодой человек 20-25 лет. "
            "Отвечай только на русском языке в повседневном неформальном тоне, других языков ты не знаешь. ТОЛЬКО РУССКИЙ ЯЗЫК. "
            "Не повторяй сообщение пользователя. Не включай метки [User] или [Bot] в ответ. "
            "Выводи только текст ответа без лишних символов."
        )
        # Базовый стоп-лист (можно расширить)
        self.blocklist = [r"\bmaliciousword\b"]
        self.blocklist_re = [re.compile(p, re.IGNORECASE) for p in self.blocklist]

    def _is_blocked(self, text: str) -> bool:
        if not text:
            return False
        return any(rx.search(text) for rx in self.blocklist_re)

    def _build_prompt(self, chat_id: int, incoming_text: str) -> str:
        # Строим контекст диалога
        conv = []
        for role, msg in self.histories[chat_id]:
            prefix = "User" if role == "user" else "Assistant"
            conv.append(f"{prefix}: {msg}")
        conv.append(f"User: {incoming_text}")
        conv.append("Assistant:")
        prompt = self.system_instruction + "\n\n" + "\n".join(conv)
        return prompt

    def _clean_reply(self, raw: str, user_text: str) -> str:
        if not raw:
            return ""
        # Удаляем возможные метки в начале ответа
        raw = re.sub(r'^\s*(?:Assistant|Bot|\[Bot\]|\[Assistant\])[:\-\s]*', '', raw, flags=re.IGNORECASE)
        # Обрезаем всё после маркеров продолжения диалога
        cut_markers = ("\nUser:", "\n[User]:", "\nAssistant:", "\n[Assistant]:", "\nBot:", "\n[Bot]:")
        for marker in cut_markers:
            idx = raw.find(marker)
            if idx != -1:
                raw = raw[:idx]
        # Удаляем повторяющиеся подряд строки
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        dedup = []
        prev = None
        repeat_count = 0
        for ln in lines:
            if ln == prev:
                repeat_count += 1
                if repeat_count > 2:
                    continue
            else:
                repeat_count = 0
            dedup.append(ln)
            prev = ln
        clean = "\n".join(dedup).strip()
        # Если ответ очень похож на пользовательское сообщение (эхо), возвращаем пустую строку
        ut = user_text.strip().lower()
        ct = clean.strip().lower()
        if not ct:
            return ""
        if len(ct.split()) <= 6 and (ct == ut or ct.startswith(ut) or ut.startswith(ct)):
            return ""
        return clean

    async def handle_new_message(self, event):
        """
        Обработчик новых сообщений Telegram.
        """
        # Получаем текст и идентификатор чата
        message = getattr(event, "message", event)
        text = getattr(message, "text", None) or getattr(message, "message", None) or getattr(event, "raw_text", None)
        if not text:
            return
        chat_id = getattr(message, "chat_id", None) or getattr(message, "peer_id", None) or getattr(message, "sender_id", None)
        # Не отвечаем сами себе
        me = await self.tele.client.get_me()
        sender_id = getattr(message, "sender_id", None)
        if sender_id == me.id:
            return
        # Игнорируем команды
        if text.strip().startswith("/"):
            logger.debug("Получена команда, пропускаем")
            return
        # Блокируем нежелательные сообщения
        if self._is_blocked(text):
            logger.warning("Сообщение соответствует стоп-листу, пропускаем")
            return

        async with self.semaphore:
            try:
                logger.info(f"Обрабатываем сообщение из чата {chat_id}: {text[:50]!r}...")
                prompt = self._build_prompt(chat_id, text)
                reply_raw = await self.llm.generate(prompt, max_new_tokens=150)
                reply_text = self._clean_reply(reply_raw, user_text=text)
                if not reply_text:
                    logger.info("LLM вернул пустой или повторяющийся ответ, пропускаем")
                    return
                # Сохраняем историю диалога
                self.histories[chat_id].append(("user", text))
                self.histories[chat_id].append(("bot", reply_text))
                # Отправляем ответ
                await self.tele.send_message(target=chat_id, text=reply_text)
                logger.info(f"Отправлен ответ в чат {chat_id}")
            except Exception as e:
                logger.exception(f"Ошибка при генерации/отправке ответа: {e}")


async def run_main():
    """
    Основная асинхронная функция запуска бота.
    """
    tele_client = TelethonApiClient(
        max_retries=int(os.getenv("TELE_MAX_RETRIES", "5")),
        retry_delay=int(os.getenv("TELE_RETRY_DELAY", "5"))
    )
    llm = create_llm_from_env()
    concurrency = int(os.getenv("LLM_CONCURRENCY", "1"))
    max_history = int(os.getenv("LLM_MAX_HISTORY", "12"))

    async with tele_client:
        agent = TelegramLLMAgent(tele_client, llm, concurrency=concurrency, max_history=max_history)
        from telethon import events
        tele_client.client.add_event_handler(agent.handle_new_message, events.NewMessage(incoming=True))
        logger.info("LLM Telegram agent запущен. Ожидаем сообщения...")
        stop_event = asyncio.Event()
        # Ожидаем сигнал останова
        for signame in ("SIGINT", "SIGTERM"):
            try:
                asyncio.get_running_loop().add_signal_handler(getattr(__import__("signal"), signame),
                                                            lambda: stop_event.set())
            except NotImplementedError:
                pass
        await stop_event.wait()


def main():
    """
    Входная точка. Запускаем асинхронный цикл.
    """
    try:
        asyncio.run(run_main())
    except KeyboardInterrupt:
        logger.info("Остановлено вручную.")


if __name__ == "__main__":
    main()
