# main.py
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
WEBHOOK_HOST = os.getenv("WEBHOOK_HOST", "https://your-render-url.onrender.com")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = f"{WEBHOOK_HOST}{WEBHOOK_PATH}"

MODEL = "google/gemma-1.1-2b-it"  # или gemma-1.1-7b-it

# === Инициализация ===
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
router = Router()

# === Формат запроса к HF Inference API ===
def query_gemma(prompt: str) -> str:
    API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.95,
            "do_sample": True
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        # У Gemma ответ приходит в виде списка
        generated_text = result[0]["generated_text"]
        # Убираем дублирование входного промпта
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()
    else:
        error_msg = f"Ошибка HF API: {response.status_code} - {response.text}"
        print(error_msg)
        return "Извините, произошла ошибка при генерации ответа."

# === Обработчики ===
@router.message(Command("start"))
async def send_welcome(message: Message):
    await message.answer("Привет! Напиши мне что-нибудь, и я отвечу с помощью Gemma от Google 🤖")

@router.message()
async def handle_message(message: Message):
    user_text = message.text
    await message.answer("Думаю... ⏳")
    response = query_gemma(user_text)
    await message.answer(response)

dp.include_router(router)

# === Webhook setup ===
async def on_startup(app: web.Application):
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown(app: web.Application):
    await bot.delete_webhook()

if __name__ == "__main__":
    app = web.Application()
    webhook_requests_handler = SimpleRequestHandler(dispatcher=dp, bot=bot)
    webhook_requests_handler.register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))