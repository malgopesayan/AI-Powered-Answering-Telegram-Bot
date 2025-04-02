# AI-Powered-Answering-Telegram-Bot
An AI-powered Telegram bot that fetches MCQ answers from Gemini, NVIDIA, LLaMA, DeepSeek, and ChatGPT within 13 seconds. It uses asynchronous processing for fast and accurate responses, ensuring seamless user interaction. Ideal for quick knowledge retrieval and diverse AI-generated insights. 🚀
# AI-Powered MCQ Answering Telegram Bot

## 📌 Overview
This Telegram bot provides **instant answers** to multiple-choice questions (MCQs) by fetching responses from **five advanced AI models**: 
**Gemini, NVIDIA, LLaMA, DeepSeek, and ChatGPT**. The bot ensures accurate and diverse responses within **13 seconds**.

## 🚀 Features
- **Multi-AI Integration:** Fetches answers from **five AI models** for varied insights.
- **Fast Response Time:** Optimized to deliver results **within 13 seconds**.
- **Telegram Integration:** Seamless interaction via a Telegram bot.
- **MCQ-Based Queries:** Designed specifically for answering multiple-choice questions.

## 🛠️ Technologies Used
- **Python** (Core logic & API integration)
- **Telegram Bot API** (For communication with users)
- **AI APIs** (Gemini, NVIDIA, LLaMA, DeepSeek, ChatGPT)
- **Async Programming** (To optimize API request handling)

## 📦 Installation & Setup
1. **Clone the repository:**  
   ```sh
   gh repo clone malgopesayan/AI-Powered-Answering-Telegram-Bot
   cd aAI-Powered-Answering-Telegram-bot
   ```
2. **Install dependencies:**  
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**  
   - Get API keys for **Gemini, NVIDIA, LLaMA, DeepSeek, and ChatGPT**.
   - Set them in a `.env` file:
   ```env
   GEMINI_API_KEY=your_key
   NVIDIA_API_KEY=your_key
   LLAMA_API_KEY=your_key
   DEEPSEEK_API_KEY=your_key
   CHATGPT_API_KEY=your_key
   TELEGRAM_BOT_TOKEN=your_token
   ```
4. **Run the bot:**  
   ```sh
   python bot.py
   ```

## 🏗️ How It Works
1. User sends an **MCQ-based question** to the Telegram bot.
2. The bot **queries five AI models** asynchronously.
3. AI responses are collected and processed.
4. The **final answers** are sent back to the user within **13 seconds**.

## 📸 Screenshots
*(Add screenshots of your bot in action!)*

## 🤝 Contributing
Want to improve this project? Follow these steps:
1. **Fork** the repository.
2. **Create** a new branch.
3. **Make changes** and commit.
4. **Open a pull request**.

## 📜 License
This project is licensed under the **MIT License**.

## 📬 Contact
For any queries, feel free to reach out:
- **Email:** malgopesayan19@gmail.com  
- **LinkedIn:** [Sayan Malgope](https://www.linkedin.com/in/malgopesayan/)
