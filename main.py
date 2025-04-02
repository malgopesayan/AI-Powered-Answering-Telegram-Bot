import os
import logging
import re
import tempfile
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    filters
)
from PIL import Image
import google.generativeai as genai
from openai import OpenAI
import groq

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# API configuration
API_KEYS = {
    "GEMINI_ANSWER": os.getenv("GEMINI_ANSWER", "Gemini_api_key"),
    "GEMINI_TEXT": os.getenv("GEMINI_TEXT", "Gemini_api_key"),
    "NVIDIA": os.getenv("NVIDIA", "Nvidia_api_key"),
    "GROQ": os.getenv("GROQ", "Groq_api_key_for_deepseek"),
    "GITHUB": os.getenv("GITHUB", "github_api_key_chat_gpt")
}

# Initialize AI clients
groq_client = groq.Client(api_key=API_KEYS["GROQ"])
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEYS["NVIDIA"]
)
azure_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=API_KEYS["GITHUB"]
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message"""
    welcome_text = (
        "🤖 Welcome to AI MCQ Solver Bot!\n\n"
        "Send me a photo of your MCQ question and I'll analyze it with multiple AI models!\n\n"
        "Supported formats: JPEG, PNG\n"
        "Max size: 5MB"
    )
    await update.message.reply_text(welcome_text)

async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process incoming images with immediate responses"""
    message = update.message
    photo = message.photo[-1] if message.photo else None
    
    if not photo:
        await message.reply_text("Please send a proper image file")
        return

    img_path = None
    try:
        # Download image
        file = await context.bot.get_file(photo.file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            await file.download_to_memory(f)
            img_path = f.name

        # Create initial processing message
        status_msg = await message.reply_text("🔍 Processing with AI models...")
        
        # Start parallel tasks
        gemini_task = asyncio.create_task(gemini_answer(img_path))
        text_extract_task = asyncio.create_task(gemini_text_extract(img_path))
        
        try:
            # Get extracted text for other models
            text = await text_extract_task
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            text = None

        # Prepare model tasks
        tasks = [gemini_task]
        text_models = [
            ("NVIDIA", "nvidia"),
            ("Llama", "llama"),
            ("GPT-4o", "gpt4o"),
            ("Deepseek", "deepseek"),
        ]

        if text:
            # Create text-based model tasks
            for model_name, model_key in text_models:
                task = asyncio.create_task(
                    process_model_with_name(model_name, text, model_key)
                )
                tasks.append(task)

        # Process and send answers as they arrive
        for future in asyncio.as_completed(tasks):
            model_name, result = await future
            await message.reply_text(f"✦ {model_name}:\n{result}")

        # Delete initial status message
        await status_msg.delete()

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await message.reply_text("❌ Error processing your question")
    finally:
        if img_path and os.path.exists(img_path):
            os.unlink(img_path)

async def gemini_answer(img_path):
    """Get answer from Gemini Vision"""
    try:
        genai.configure(api_key=API_KEYS["GEMINI_ANSWER"])
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([
                "Give only the correct answer to this question, no explanation, like 'the correct answer is: a)19'",
                img
            ])
        return ("Gemini", clean_response(response.text))
    except Exception as e:
        return ("Gemini", f"Error: {str(e)}")

async def process_model_with_name(model_name, text, model_key):
    """Process model with error handling and return tuple"""
    try:
        prompt = f"""Question: {text}
        please  Give only the correct answer to this question, no explanation, like 'the correct answer is: a)19'"""
        
        if model_key == "nvidia":
            response = nvidia_client.chat.completions.create(
                model="nvidia/nemotron-4-340b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
        elif model_key == "llama":
            response = groq_client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
        elif model_key == "gpt4o":
            response = azure_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
        elif model_key == "deepseek":
            response = groq_client.chat.completions.create(
                model="deepseek-r1-distill-llama-70b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            
        return (model_name, clean_response(response.choices[0].message.content))
    
    except Exception as e:
        return (model_name, f"API Error: {str(e)}")

async def gemini_text_extract(img_path):
    """Extract text using Gemini"""
    try:
        genai.configure(api_key=API_KEYS["GEMINI_TEXT"])
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([
                "Extract all text exactly as it appears in the image",
                img
            ])
        return response.text
    except Exception as e:
        raise Exception(f"Text Extraction Failed: {str(e)}")

def clean_response(text):
    """Clean model response"""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"Correct Answer:\s*(.*)", cleaned, re.IGNORECASE)
    return match.group(0).strip() if match else cleaned.strip()

def main():
    """Start the bot"""
    application = Application.builder().token("Telegram_token").build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))
    application.add_handler(MessageHandler(filters.Document.IMAGE, handle_image))

    # Run the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
