import os
import requests
from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
import logging

# === Настройки ===
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Получаем домен из переменной окружения (надёжнее, чем RENDER_EXTERNAL_URL)
webhook_domain = os.getenv("WEBHOOK_DOMAIN")
if webhook_domain:
    WEBHOOK_URL = f"https://{webhook_domain}/webhook"
else:
    raise ValueError("Переменная WEBHOOK_DOMAIN не задана в Render!")

# Проверки на токены
if not OPENROUTER_TOKEN:
    raise ValueError("OPENROUTER_TOKEN не задан!")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN не задан!")

# === Инициализация ===
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()
router = Router()

# === Запрос к OpenRouter ===
def query_qwen(prompt: str) -> str:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_TOKEN}"}
    payload = {
        "model": "qwen/qwen2.5-vl-32b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.95
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Ошибка OpenRouter: {response.status_code} - {response.text[:200]}")
            return "Не удалось получить ответ от модели."
    except Exception as e:
        print(f"Исключение: {str(e)}")
        return "Ошибка при обращении к модели."

# === Обработчики ===
@router.message(Command("start"))
async def send_welcome(message: Message):
    await message.answer("Привет! Я бот на базе Qwen от Alibaba 🤖\nНапишите мне что-нибудь!")

@router.message()
async def handle_message(message: Message):
    user_text = message.text
    thinking_msg = await message.answer("Думаю... ⏳")
    response = query_qwen(user_text)
    await thinking_msg.edit_text(response)

dp.include_router(router)

# === Webhook setup ===
async def on_startup(app: web.Application):
    await bot.set_webhook(WEBHOOK_URL)
    print(f"Webhook установлен: {WEBHOOK_URL}")

async def on_shutdown(app: web.Application):
    await bot.delete_webhook()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", 10000))
    app = web.Application()
    webhook_requests_handler = SimpleRequestHandler(dispatcher=dp, bot=bot)
    webhook_requests_handler.register(app, path="/webhook")
    setup_application(app, dp, bot=bot)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=port)