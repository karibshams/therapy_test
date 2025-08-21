import streamlit as st
import os
import asyncio
from datetime import datetime
from main import EmothriveAI, EmothriveBackendInterface
from voice_input import RealTimeVoiceInput
import time

st.set_page_config(
    page_title="Emothrive AI Therapist Chat", 
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
.voice-indicator {
    background-color: #ff4b4b;
    color: white;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.voice-status {
    background-color: #00d4aa;
    color: white;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 14px;
    margin: 5px 0;
}

.transcript-box {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #ff4b4b;
    margin: 10px 0;
    min-height: 50px;
}

.chat-message {
    padding: 10px;
    margin: 5px 0;
    border-radius: 10px;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: 20px;
}

.assistant-message {
    background-color: #f3e5f5;
    margin-right: 20px;
}

.voice-message {
    border-left: 4px solid #ff4b4b;
}
</style>
""", unsafe_allow_html=True)

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
if "processing_voice" not in st.session_state:
    st.session_state.processing_voice = False
if "last_transcript_update" not in st.session_state:
    st.session_state.last_transcript_update = 0

# Voice callback functions
def on_transcript_update(transcript):
    """Called when transcript is updated in real-time"""
    st.session_state.current_transcript = transcript
    st.session_state.last_transcript_update = time.time()
    # Force UI update
    if hasattr(st.session_state, '_transcript_placeholder'):
        st.session_state._transcript_placeholder.text_input(
            "🎤 Live Transcript", 
            value=transcript, 
            key=f"transcript_{st.session_state.last_transcript_update}",
            disabled=True
        )

def on_final_transcript(transcript):
    """Called when final transcript is ready"""
    st.session_state.processing_voice = True
    
    if transcript.strip():
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": transcript,
            "timestamp": datetime.now(),
            "source": "voice"
        })
        
        # Process with AI
        process_ai_message(transcript, source="voice")
    
    # Reset states
    st.session_state.is_recording = False
    st.session_state.current_transcript = ""
    st.session_state.processing_voice = False
    
    # Force rerun
    st.rerun()

def on_recording_start():
    """Called when recording starts"""
    st.session_state.is_recording = True
    st.session_state.current_transcript = ""
    st.session_state.processing_voice = False

def on_recording_stop():
    """Called when recording stops"""
    st.session_state.is_recording = False

# Initialize AI engine
@st.cache_resource
def initialize_ai_engine():
    """Initialize AI engine (cached to prevent reinitialization)"""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_key:
        st.error("OpenAI API Key missing. Please set in .env file or environment variable.")
        return None, None
    
    try:
        ai_engine = EmothriveAI(openai_api_key=openai_key)
        backend = EmothriveBackendInterface(ai_engine=ai_engine)
        return ai_engine, backend
    except Exception as e:
        st.error(f"Failed to initialize AI engine: {e}")
        return None, None

def initialize_voice_system():
    """Initialize voice system with callbacks"""
    try:
        if st.session_state.voice_system is None:
            voice_system = RealTimeVoiceInput()
            voice_system.set_callbacks(
                on_transcript_update=on_transcript_update,
                on_final_transcript=on_final_transcript,
                on_recording_start=on_recording_start,
                on_recording_stop=on_recording_stop
            )
            st.session_state.voice_system = voice_system
            st.session_state.voice_error = None
            return True
    except Exception as e:
        st.session_state.voice_error = str(e)
        st.session_state.voice_system = None
        st.error(f"Voice system initialization failed: {e}")
        return False

# Process AI message
async def get_ai_response(user_message: str, source="text"):
    """Get AI response"""
    request = {"message": user_message, "source": source}
    ai_engine, backend = initialize_ai_engine()
    if backend:
        return await backend.process_message(request)
    return {"success": False, "error": "AI engine not initialized"}

def process_ai_message(user_message, source="text"):
    """Process message with AI and add response to history"""
    try:
        # Create new event loop for async call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(get_ai_response(user_message, source))
            
            if response.get("success"):
                response_text = response["response"]["text"]
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now(),
                    "source": "ai"
                })
                
                # Show success message for voice input
                if source == "voice":
                    st.success("✅ Voice message processed successfully!")
                    
            else:
                error_msg = f"I'm sorry, I encountered an error: {response.get('error', 'Unknown error')}"
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": error_msg,
                    "timestamp": datetime.now(),
                    "source": "ai"
                })
        finally:
            loop.close()
            
    except Exception as e:
        error_msg = f"I'm sorry, I encountered an error processing your message: {str(e)}"
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": error_msg,
            "timestamp": datetime.now(),
            "source": "ai"
        })
        st.error(f"Processing error: {str(e)}")

def display_conversation():
    """Display conversation history"""
    st.subheader("💬 Conversation History")
    
    if not st.session_state.conversation_history:
        st.info("Start a conversation by typing or speaking...")
        return
    
    for i, message in enumerate(st.session_state.conversation_history):
        timestamp = message.get('timestamp', datetime.now()).strftime("%H:%M:%S")
        source = message.get('source', 'text')
        
        if message["role"] == "user":
            source_icon = "🎤" if source == "voice" else "💬"
            class_name = "chat-message user-message voice-message" if source == "voice" else "chat-message user-message"
            st.markdown(f"""
            <div class="{class_name}">
                <strong>{source_icon} You ({timestamp}):</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🧠 Therapist ({timestamp}):</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

def main():
    st.title("🧠 Emothrive AI Therapist Chat")
    
    # Initialize systems
    ai_engine, backend = initialize_ai_engine()
    
    if ai_engine is None or backend is None:
        st.error("Failed to initialize AI system. Please check your OpenAI API key.")
        st.stop()
    
    # Store in session state
    if not st.session_state.initialized:
        st.session_state.ai_engine = ai_engine
        st.session_state.backend_interface = backend
        st.session_state.initialized = True
        
        # Initialize voice system
        if initialize_voice_system():
            st.success("✅ AI Engine and Voice System initialized successfully!")
        else:
            st.warning("⚠️ AI Engine initialized, but voice system failed. Text input only.")
    
    # Voice Control Section
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.session_state.voice_system:
            if st.button("🎤 Start Voice Recording" if not st.session_state.is_recording else "🛑 Stop Voice Recording", 
                        type="primary" if not st.session_state.is_recording else "secondary"):
                if st.session_state.is_recording:
                    st.session_state.voice_system.stop_recording()
                    st.session_state.is_recording = False
                else:
                    st.session_state.voice_system.start_recording()
        else:
            st.button("🎤 Voice Unavailable", disabled=True)
    
    with col2:
        if st.session_state.voice_error:
            st.error(f"Voice Error: {st.session_state.voice_error}")
        elif st.session_state.is_recording:
            st.markdown('<div class="voice-status">🎤 Recording... Speak now</div>', unsafe_allow_html=True)
        elif st.session_state.processing_voice:
            st.markdown('<div class="voice-status">🔄 Processing voice...</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Reset Chat"):
            st.session_state.conversation_history = []
            st.session_state.current_transcript = ""
            st.rerun()
    
    # Live Transcript Display
    if st.session_state.is_recording or st.session_state.current_transcript:
        st.markdown("### 🎤 Live Voice Input")
        transcript_container = st.container()
        with transcript_container:
            if st.session_state.current_transcript:
                st.markdown(f"""
                <div class="transcript-box">
                    <strong>Current Speech:</strong><br>
                    {st.session_state.current_transcript}
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.is_recording:
                st.markdown(f"""
                <div class="transcript-box">
                    <strong>Listening...</strong><br>
                    <em>Start speaking...</em>
                </div>
                """, unsafe_allow_html=True)
    
    # Text Input Section
    st.markdown("---")
    st.markdown("### ✍️ Text Input")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            key="chat_input",
            placeholder="Type your message or use voice input above...",
            disabled=st.session_state.is_recording
        )
    
    with col2:
        send_button = st.button("Send", type="primary", disabled=st.session_state.is_recording)
    
    # Process text input
    if (send_button or user_input) and user_input and not st.session_state.is_recording:
        # Add to conversation history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(),
            "source": "text"
        })
        
        # Process with AI
        with st.spinner("🤔 Thinking..."):
            process_ai_message(user_input, source="text")
        
        # Clear input and rerun
        st.session_state.chat_input = ""
        st.rerun()
    
    # Display conversation
    st.markdown("---")
    display_conversation()
    
    # Auto-refresh for voice updates
    if st.session_state.is_recording:
        time.sleep(0.5)
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)