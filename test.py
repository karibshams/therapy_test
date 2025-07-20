import streamlit as st
import os
import asyncio
from datetime import datetime
from main import EmothriveAI, EmothriveBackendInterface

st.set_page_config(page_title="Emothrive AI Chat", page_icon="🧠")

def initialize_ai_engine():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        st.error("OpenAI API Key missing. Please set in .env file or environment variable.")
        return False
    
    ai_engine = EmothriveAI(openai_api_key=openai_key)
    backend = EmothriveBackendInterface(ai_engine=ai_engine)
    
    st.session_state.ai_engine = ai_engine
    st.session_state.backend_interface = backend
    st.session_state.conversation_history = []
    st.session_state.initialized = True
    return True

async def get_ai_response(user_message: str):
    request = {"message": user_message}
    return await st.session_state.backend_interface.process_message(request)

def main():
    st.title("🧠 Emothrive AI Therapist Chat")

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    if not st.session_state.initialized:
        if st.button("Initialize AI Engine"):
            initialize_ai_engine()
        st.stop()

    user_input = st.text_input("Type your message here...", key="chat_input")
    if user_input:
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(get_ai_response(user_input))

        if response.get("success"):
            # Process response
            response_text = response["response"]["text"]
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now()
            })

        # Display conversation history
        for msg in st.session_state.conversation_history:
            if msg["role"] == "user":
                st.markdown(f"<div style='text-align: right; background:#DCF8C6; padding:10px; margin:5px; border-radius:10px;'>**You:** {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; background:#F1F0F0; padding:10px; margin:5px; border-radius:10px;'>**Therapist:** {msg['content']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
