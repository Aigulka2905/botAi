import os
import requests
from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
import logging
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util

# === Настройки ===
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Получаем домен из переменной окружения
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

# === Загрузка инструкций из PDF (RAG setup) ===
# Укажите пути к вашим PDF-файлам (загрузите их в корень проекта)
PDF_FILES = ["instructions.pdf"]  # Добавьте все файлы, если несколько

# Модель для embeddings (локальная, не требует API)
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Лёгкая модель для семантики

# Извлекаем текст из PDF и разбиваем на абзацы
documents = []
for pdf_path in PDF_FILES:
    if os.path.exists(pdf_path):
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]  # Разбиваем на абзацы
                documents.extend(paragraphs)
    else:
        logging.warning(f"PDF файл {pdf_path} не найден!")

# Предвычисляем embeddings для документов
if documents:
    doc_embeddings = embedder.encode(documents, convert_to_tensor=True)
else:
    doc_embeddings = None
    logging.error("Нет документов для RAG!")

# === Запрос к OpenRouter с RAG ===
def query_qwen(prompt: str) -> str:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_TOKEN}"}

    # RAG: Поиск релевантных документов
    relevant_context = ""
    if doc_embeddings is not None:
        query_embedding = embedder.encode(prompt, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        top_results = cos_scores.topk(3)  # Топ-3 релевантных абзаца
        for score, idx in zip(top_results[0], top_results[1]):
            if score > 0.5:  # Порог релевантности (настройте)
                relevant_context += documents[idx] + "\n\n"

    # Формируем messages с контекстом, если он есть
    messages = [
        {"role": "system", "content": "Ты эксперт по электронным торговым площадкам (ЭТП). Отвечай на вопросы пользователя пошагово, четко и только на русском языке. Если вопрос не касается ЭТП, скажи 'Это вне моей специализации'."}
    ]
    if relevant_context:
        messages.append({"role": "system", "content": f"Используй этот контекст из инструкций для ответа: {relevant_context}"})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "qwen/qwen2.5-vl-32b-instruct:free",
        "messages": messages,
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
    await message.answer("Привет! Я бот-эксперт по электронным торговым площадкам (ЭТП) на базе Qwen от Alibaba 🤖\nЗадайте вопрос по инструкциям работы на ЭТП, например: 'Как зарегистрироваться на ЕИС?'")

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