# 🤖 AI Chatbot Mentor

An AI-powered, module-based learning mentor built using **Streamlit**, **LangChain**, and **Google Gemini 2.5 Flash**.

## 🚀 Features

- 📘 Module-based mentor system  
  - Python  
  - SQL  
  - Power BI  
  - EDA  
  - Machine Learning  
  - Deep Learning  
  - Generative AI  
  - Agentic AI  

- 🧠 Context-aware memory (last 15 interactions)
- 🚫 Strict domain restriction (prevents hallucinations)
- 🎯 Module-specific welcome interface
- 💬 Full chat history rendering
- 📥 Download conversation as `.txt` (for notes & revision)
- ⚡ Powered by **Gemini 2.5 Flash**

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain (Runnable architecture)
- Google Gemini API
- Gemini 2.5 Flash
- python-dotenv

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/AI_Chatbot_Mentor.git
cd AI_Chatbot_Mentor
2️⃣ Create virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Set API key
Create a .env file:

ini
Copy code
GOOGLE_API_KEY=your_gemini_api_key_here
5️⃣ Run the app
bash
Copy code
python -m streamlit run app.py
📥 Download Feature
Users can download the full chat history in .txt format for:

Revision

Notes

Offline learning

Portfolio documentation

👨‍💻 Author
Built as a learning & portfolio project to explore:

Modern LangChain architecture

Safe AI systems

Educational AI design

yaml
Copy code

---

## 🌐 5. Create GitHub Repository

1. Go to 👉 https://github.com
2. Click **New Repository**
3. Name it:  
AI_Chatbot_Mentor

markdown
Copy code
4. Description:
Module-based AI Mentor using Streamlit, LangChain & Gemini 2.5 Flash

yaml
Copy code
5. Public ✅
6. **DO NOT** initialize with README (you already have one)
7. Click **Create repository**

---

## 🚀 6. Push Code to GitHub (Commands)

Open terminal **inside your project folder**:

```bash
git init
git add .
git commit -m "Initial commit: AI Chatbot Mentor with Gemini 2.5 Flash"
Add GitHub remote (replace username):

bash
Copy code
git branch -M main
git remote add origin https://github.com/<your-username>/AI_Chatbot_Mentor.git
git push -u origin main
✅ 7. Verify on GitHub
Your repo should now show:

✅ app.py

✅ README.md

✅ requirements.txt

❌ .env (not visible)

🏆 You Now Have a STRONG Portfolio Project
This project demonstrates:

Real-world debugging

Modern LLM architecture

Safe AI design (domain restriction)

UX thinking (chat history + download)

Production-ready practices