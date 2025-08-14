import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "gpt2"          # Должен совпадать с тем, что ты использовал при обучении
LORA_DIR = r"D:\projects\python\TelegramGraphNet\TeleNet\lora_out"        # Путь к папке с LoRA-адаптером

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Загружаем базовую модель
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch_dtype, device_map="auto")

# Подключаем LoRA-адаптер
model = PeftModel.from_pretrained(model, LORA_DIR)

# Генерация
while True:
    prompt = input("You: ")
    if not prompt.strip():
        break
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    print("Bot:", tokenizer.decode(outputs[0], skip_special_tokens=True))
