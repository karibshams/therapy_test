Here’s a **professional README.md** you can use for your `EmothriveAI` project, based directly on the code you shared:

---

# EmothriveAI 🎧🧠

**AI-Powered Therapeutic Voice Assistant with PDF Knowledge Base**

---

## 📌 Overview

EmothriveAI is an **AI-driven therapeutic assistant** designed to support mental wellness through empathetic conversations.
It combines **LLM-powered therapy**, **voice input/output**, and **PDF-based knowledge retrieval** to create a warm and supportive experience.

Key Features:

* 🎙️ **Voice Input & Transcription** (speech-to-text)
* 🔊 **Therapeutic Voice Output** with adaptive speech styles
* 📚 **PDF Knowledge Base** for contextual therapy guidance
* 🧠 **Therapy-Specific Prompts** and empathetic response tuning
* ⚡ **Real-Time Conversation Handling** with OpenAI GPT models
* 🛡️ **Crisis Detection Support** (optional safeguard mode)

---

## 🏗️ Project Structure

```
project/
│── app.py                # Main AI engine (EmothriveAI + Backend Interface)
│── pdf_processor.py       # Handles PDF ingestion, embeddings, and vector search
│── prompt.py              # Therapy types, prompt management, and conversation style
│── finalvoice.py          # Voice input (speech-to-text)
│── voiceoutput.py         # Voice output (text-to-speech + therapeutic voice manager)
│── .env                   # Environment variables (API keys, configs)
│── pdf/                   # Folder for knowledge-base PDFs
│── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/emothrive-ai.git
cd emothrive-ai
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file in the project root:

```ini
OPENAI_API_KEY=your_openai_api_key_here
AZURE_TTS_KEY=your_azure_tts_api_key_here
AZURE_TTS_REGION=eastus   # Example region
```

---

## 🚀 Usage

### Running AI Engine

```python
from app import EmothriveAI, EmothriveBackendInterface
from prompt import TherapyType

ai_engine = EmothriveAI(openai_api_key="your_api_key_here")
backend = EmothriveBackendInterface(ai_engine)

# Example: Process a text message
import asyncio
result = asyncio.run(backend.process_message({"message": "I feel anxious today"}))
print(result)
```

### Voice Input

```python
result = asyncio.run(backend.process_message({
    "is_voice_input": True,
    "enable_voice_output": True
}))
print(result)
```

---

## 🎯 Features in Detail

* **PDF Knowledge Base**

  * Upload PDFs to `./pdf/`
  * Automatically indexed and vectorized for context retrieval

* **Prompt Management**

  * Therapy types (e.g., CBT, General Support, Mindfulness)
  * Empathetic conversation style by default

* **Voice System**

  * `finalvoice.py` → Speech-to-text
  * `voiceoutput.py` → Text-to-speech with therapeutic tones

* **Session Tracking**

  * Session ID
  * Start time
  * Message counts
  * Therapy type usage

---

## 🧩 Example Response Flow

1. User speaks → `VoiceInput` transcribes speech
2. Message processed → Prompt + PDF context applied
3. LLM generates response → `_make_warm_and_supportive()` adjusts tone
4. Response stored in conversation history
5. If enabled → `VoiceOutput` delivers response with empathetic voice

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **OpenAI GPT (Chat Completions API)**
* **Azure Cognitive Services (TTS)**
* **PDF Vector Store (FAISS / embeddings)**
* **Asyncio** for real-time conversation flow

---

## 🧹 Cleanup

When shutting down:

```python
backend.cleanup()
```

---

## 📌 Roadmap

* [ ] Mood-adaptive conversation flow improvements
* [ ] Crisis detection with escalation protocols
* [ ] Web/Streamlit interface for therapy sessions
* [ ] Multi-language voice support

---

## ⚖️ Disclaimer

This project is **for educational and supportive purposes only**.
It is **not a replacement for professional therapy or medical advice**.
If you are experiencing a crisis, please seek immediate help from a licensed professional or emergency services.

---

Would you like me to also create a **`requirements.txt`** (dependencies list) for this code so it’s ready to run without missing packages?


