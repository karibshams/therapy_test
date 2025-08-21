import streamlit as st
import os
import asyncio
from datetime import datetime
from main import EmothriveAI, EmothriveBackendInterface
from voice_input import RealTimeVoiceInput

st.set_page_config(page_title="Emothrive AI Therapist Chat", page_icon="🧠")

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "current_transcript" not in st.session_state:
    st.session_state.current_transcript = ""
if "voice_system" not in st.session_state:
    st.session_state.voice_system = None
if "voice_error" not in st.session_state:
    st.session_state.voice_error = None

# Voice callback functions
def on_transcript_update(transcript):
    """Called when transcript is updated in real-time"""
    st.session_state.current_transcript = transcript
    st.rerun()  # Trigger UI update

def on_final_transcript(transcript):
    """Called when final transcript is ready"""
    if transcript.strip():
        st.session_state.conversation_history.append({
            "role": "user",
            "content": transcript,
            "timestamp": datetime.now(),
            "source": "voice"
        })
        process_ai_message(transcript, source="voice")
    st.session_state.is_recording = False
    st.session_state.current_transcript = ""
    st.rerun()

def on_recording_start():
    st.session_state.is_recording = True
    st.session_state.current_transcript = ""
    st.rerun()

def on_recording_stop():
    st.session_state.is_recording = False
    st.rerun()

# Initialize AI engine
def initialize_ai_engine():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        st.error("OpenAI API Key missing. Please set in .env file or environment variable.")
        return False
    
    try:
        ai_engine = EmothriveAI(openai_api_key=openai_key)
        backend = EmothriveBackendInterface(ai_engine=ai_engine)
        
        st.session_state.ai_engine = ai_engine
        st.session_state.backend_interface = backend
        st.session_state.conversation_history = []
        st.session_state.initialized = True
        
        initialize_voice_system()
        st.success("✅ AI Engine and Voice System initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to initialize AI engine: {e}")
        return False

def initialize_voice_system():
    """Initialize voice system"""
    try:
        voice_system = RealTimeVoiceInput()
        voice_system.set_callbacks(
            on_transcript_update=on_transcript_update,
            on_final_transcript=on_final_transcript,
            on_recording_start=on_recording_start,
            on_recording_stop=on_recording_stop
        )
        st.session_state.voice_system = voice_system
        st.session_state.voice_error = None
    except Exception as e:
        st.session_state.voice_error = str(e)
        st.session_state.voice_system = None

# Process AI message
async def get_ai_response(user_message: str):
    request = {"message": user_message}
    return await st.session_state.backend_interface.process_message(request)

def process_ai_message(user_message, source="text"):
    """Process message with AI and add response to history"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        response = loop.run_until_complete(get_ai_response(user_message))

        if response.get("success"):
            response_text = response["response"]["text"]
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now()
            })
        else:
            st.session_state.conversation_history.append({
                "role": "assistant", 
                "content": f"I'm sorry, I encountered an error: {response.get('error', 'Unknown error')}",
                "timestamp": datetime.now()
            })
    except Exception as e:
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": f"I'm sorry, I encountered an error processing your message: {str(e)}",
            "timestamp": datetime.now()
        })
    finally:
        st.rerun()

# Main function
def main():
    st.title("🧠 Emothrive AI Therapist Chat")

    if not st.session_state.initialized:
        st.info("👋 Welcome! Please initialize the AI system.")
        if st.button("🚀 Initialize AI Engine", type="primary"):
            with st.spinner("Initializing AI and Voice systems..."):
                initialize_ai_engine()
        st.stop()

    # Display voice status
    if st.session_state.is_recording:
        st.markdown('<div class="voice-indicator">🎤 Listening...</div>', unsafe_allow_html=True)

    # Display current transcript
    st.text_input("Speak or type your message", value=st.session_state.current_transcript, key="input_field", disabled=True)

    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key="chat_input",
            placeholder="Type or speak your message..."
        )
    
    with col2:
        if st.button("🎤 Speak"):
            if st.session_state.voice_system:
                if st.session_state.is_recording:
                    st.session_state.voice_system.stop_recording()
                else:
                    st.session_state.voice_system.start_recording()

    if user_input:
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(),
            "source": "text"
        })
        process_ai_message(user_input, source="text")

if __name__ == "__main__":
    main()
