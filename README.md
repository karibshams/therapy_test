Here’s a polished and beautifully formatted version of your project description for **Emothrive AI Therapy Chatbot** — ideal for your README or documentation:

---

# 🌿 Emothrive AI Therapy Chatbot

**Emothrive AI** is a compassionate, AI-powered conversational therapist designed to provide supportive, evidence-based mental health guidance. Leveraging advanced NLP and clinical knowledge extracted from curated therapy PDFs, Emothrive is your AI companion for mental wellness.

---

## 🧠 Project Overview

Emothrive AI is a multi-modal, RAG-enabled (Retrieval-Augmented Generation) chatbot that integrates:

* ✅ **Therapeutic Approaches**:
  Cognitive Behavioral Therapy (CBT), Dialectical Behavioral Therapy (DBT), Acceptance and Commitment Therapy (ACT), and more.

* 📄 **PDF Knowledge Integration**:
  Learns from clinical PDFs using FAISS vector store and embeddings for accurate, context-rich responses.

* 🧘‍♀️ **Contextual Therapy Awareness**:
  Detects therapy themes like anxiety, grief, parenting, trauma, and adapts support strategies with empathy.

* 🖥️ **Streamlit-Based UI**:
  Clean and interactive web interface for engaging therapy sessions.

---

## ✨ Key Features

* 🎭 **Dynamic Therapy Detection**
  Automatically identifies therapy type from user input — no manual selection needed.

* 📚 **RAG-Powered Replies**
  Enriches responses using real clinical knowledge embedded from PDFs.

* 🔁 **Session Memory**
  Maintains context across conversations for a natural and connected chat flow.

* 🤖 **GPT-4.1-mini Integration**
  Utilizes OpenAI's compact yet powerful model for human-like interaction.

* 📂 **Self-Managing Vector Store**
  Auto-extracts, chunks, embeds PDFs, and creates vector index using FAISS.

* 🔐 **Secure Configuration**
  All sensitive data managed securely via `.env` files.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/therapy_test.git
cd therapy_test
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # For Linux/macOS
.venv\Scripts\activate      # For Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your OpenAI API key to a `.env` file:

```ini
OPENAI_API_KEY=your_openai_api_key_here
```

Place your therapy PDFs in the `pdf/` folder.

---

## 🧪 Usage

Run the Streamlit app:

```bash
streamlit run test.py
```

Then open your browser at:
[http://localhost:8501](http://localhost:8501)

Start your conversation with **Emothrive AI** and explore mental wellness through intelligent, empathetic dialogue.

---

## 🗂️ Project Structure

| File/Folder        | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `main.py`          | Core AI logic and vector store initialization         |
| `pdf_processor.py` | PDF processing, chunking, embedding, FAISS management |
| `prompt.py`        | Prompt creation and dynamic therapy style logic       |
| `test.py`          | Streamlit UI interface                                |
| `pdf/`             | Folder for clinical therapy PDFs                      |
| `vector_store/`    | Auto-generated vector index (Git ignored)             |
| `.env`             | Environment variables (API keys, configs)             |

---

## 🤝 Contributing

We welcome contributions!
Please feel free to:

* Open issues for bugs, feedback, or feature requests.
* Submit PRs with clear and concise changes.
* Keep all API keys and sensitive information **out of commits**.

---

## 📜 License

MIT License
© 2025 Your Name

---

## 🙏 Acknowledgements

* [OpenAI](https://openai.com/) for GPT-4.1-mini
* [LangChain](https://www.langchain.com/) for vector store and prompt tools
* [Sentence Transformers](https://www.sbert.net/) for embeddings
* [Streamlit](https://streamlit.io/) for UI framework


