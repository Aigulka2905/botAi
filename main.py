import os
import requests
from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
import logging

# === Настройки ===
HF_TOKEN = os.getenv("HF_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Получаем домен из переменной окружения (надёжнее, чем RENDER_EXTERNAL_URL)
webhook_domain = os.getenv("WEBHOOK_DOMAIN")
if webhook_domain:
    WEBHOOK_URL = f"https://{webhook_domain}/webhook"
else:
    raise ValueError("Переменная WEBHOOK_DOMAIN не задана в Render!")

# === Инициализация ===
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
router = Router()

# === Запрос к Hugging Face ===
def query_gemma(prompt: str) -> str:
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and "generated_text" in result[0]:
                # Убираем входной промпт из ответа
                full_text = result[0]["generated_text"]
                if full_text.startswith(formatted_prompt):
                    return full_text[len(formatted_prompt):].strip()
                return full_text.strip()
            else:
                return "Модель вернула неожиданный формат."
        else:
            print(f"Ошибка HF API: {response.status_code} - {response.text[:200]}")
            return "Не удалось получить ответ от модели."
    except Exception as e:
        print(f"Исключение: {str(e)}")
        return "Ошибка при обращении к модели."

# === Обработчики ===
@router.message(Command("start"))
async def send_welcome(message: Message):
    await message.answer("Привет! Я бот на базе Gemma от Google 🤖\nНапишите мне что-нибудь!")

@router.message()
async def handle_message(message: Message):
    user_text = message.text
    thinking_msg = await message.answer("Думаю... ⏳")
    response = query_gemma(user_text)
    await thinking_msg.edit_text(response)

dp.include_router(router)

# === Webhook setup ===
async def on_startup(app: web.Application):
    await bot.set_webhook(WEBHOOK_URL)
    print(f"Webhook установлен: {WEBHOOK_URL}")

async def on_shutdown(app: web.Application):
    await bot.delete_webhook()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app = web.Application()
    webhook_requests_handler = SimpleRequestHandler(dispatcher=dp, bot=bot)
    webhook_requests_handler.register(app, path="/webhook")
    setup_application(app, dp, bot=bot)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=port)