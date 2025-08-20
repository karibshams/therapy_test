# test.py - Minimal Voice Integration (Your original code + real-time voice)

import streamlit as st
import os
import asyncio
from datetime import datetime
from main import EmothriveAI, EmothriveBackendInterface
from voice_input import RealTimeVoiceInput  # Only import real-time voice

st.set_page_config(page_title="Emothrive AI Therapist Chat", page_icon="🧠")

# Add some basic styling for voice features
st.markdown("""
<style>
.voice-indicator {
    position: fixed;
    top: 10px;
    right: 10px;
    background: #ff4444;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    z-index: 1000;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.transcript-preview {
    background: #e3f2fd;
    color: #1976d2;
    padding: 8px 12px;
    margin: 4px 0;
    border-radius: 8px;
    font-style: italic;
    border-left: 3px solid #2196f3;
}

.voice-message {
    background: #e8f5e8 !important;
    border-left: 4px solid #4caf50 !important;
}

.voice-controls {
    background: #f5f5f5;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state (your original logic + voice additions)
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
    # Don't rerun too frequently to avoid performance issues
    
def on_final_transcript(transcript):
    """Called when final transcript is ready"""
    if transcript.strip():
        # Add to conversation history with voice indicator
        st.session_state.conversation_history.append({
            "role": "user",
            "content": transcript,
            "timestamp": datetime.now(),
            "source": "voice"  # Mark as voice message
        })
        
        # Process with your existing AI system
        process_ai_message(transcript)
    
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

# Initialize the AI engine (your original function)
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
        
        # Initialize voice system
        initialize_voice_system()
        
        st.success("✅ AI Engine and Voice System initialized successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to initialize AI engine: {e}")
        return False

def initialize_voice_system():
    """Initialize voice system separately"""
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

# Get AI response (your original function)
async def get_ai_response(user_message: str):
    request = {"message": user_message}
    return await st.session_state.backend_interface.process_message(request)

# Process AI message
def process_ai_message(user_message):
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

# Main function (your original structure + voice integration)
def main():
    st.title("🧠 Emothrive AI Therapist Chat")

    # Initialize AI engine if needed (your original logic)
    if not st.session_state.initialized:
        st.info("👋 Welcome! Please initialize the AI system to begin your therapy session.")
        if st.button("🚀 Initialize AI Engine", type="primary"):
            with st.spinner("Initializing AI and Voice systems..."):
                initialize_ai_engine()
        st.stop()

    # Voice system status and controls
    with st.container():
        st.markdown('<div class="voice-controls">', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.voice_error:
                st.warning(f"⚠️ Voice system unavailable: {st.session_state.voice_error}")
            elif st.session_state.voice_system:
                status = "🟢 Voice system ready" if not st.session_state.is_recording else "🔴 Listening..."
                st.success(status)
            else:
                st.warning("⚠️ Voice system not initialized")
        
        with col2:
            if st.button("🔄 Retry Voice", help="Retry voice system initialization"):
                initialize_voice_system()
                st.rerun()
        
        with col3:
            if st.session_state.voice_system and st.session_state.is_recording:
                if st.button("🛑 Force Stop", help="Force stop voice recording"):
                    st.session_state.voice_system.stop_recording()
                    st.session_state.is_recording = False
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Voice recording indicator
    if st.session_state.is_recording:
        st.markdown('<div class="voice-indicator">🎤 Listening...</div>', unsafe_allow_html=True)

    # Show real-time transcript preview
    if st.session_state.current_transcript:
        st.markdown(f"""
        <div class="transcript-preview">
            🎤 Live: "{st.session_state.current_transcript}"
        </div>
        """, unsafe_allow_html=True)

    # Input section with voice button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Clear input after processing
        input_key = f"chat_input_{len(st.session_state.conversation_history)}"
        user_input = st.text_input(
            "Type your message here...", 
            key=input_key, 
            placeholder="Type or speak your message..."
        )
    
    with col2:
        # Voice button
        voice_disabled = st.session_state.voice_system is None
        voice_button_text = "🔴 Stop" if st.session_state.is_recording else "🎤 Speak"
        
        if st.button(voice_button_text, key="voice_btn", disabled=voice_disabled, 
                    help="Click to start/stop voice input" if not voice_disabled else "Voice system unavailable"):
            if st.session_state.voice_system:
                if not st.session_state.is_recording:
                    # Start recording
                    try:
                        st.session_state.voice_system.start_recording()
                        st.session_state.is_recording = True
                    except Exception as e:
                        st.error(f"Failed to start voice recording: {e}")
                else:
                    # Stop recording
                    try:
                        st.session_state.voice_system.stop_recording()
                        st.session_state.is_recording = False
                    except Exception as e:
                        st.error(f"Failed to stop voice recording: {e}")
                st.rerun()

    # Handle text input (your original logic)
    if user_input:
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(),
            "source": "text"
        })

        process_ai_message(user_input)

    # Display conversation history (your original display + voice indicators)
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        for msg in st.session_state.conversation_history:
            timestamp = msg["timestamp"].strftime("%H:%M")
            is_voice = msg.get("source") == "voice"
            
            if msg["role"] == "user":
                # Add voice indicator for voice messages
                voice_icon = "🎤" if is_voice else "💬"
                css_class = "voice-message" if is_voice else ""
                
                st.markdown(
                    f'<div class="{css_class}" style="text-align: right; background:#DCF8C6; padding:10px; margin:5px; border-radius:10px;">'
                    f'<strong>{voice_icon} You</strong> <small>({timestamp})</small><br>'
                    f'{msg["content"]}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="text-align: left; background:#F1F0F0; padding:10px; margin:5px; border-radius:10px;">'
                    f'<strong>🧠 Therapist</strong> <small>({timestamp})</small><br>'
                    f'{msg["content"]}</div>', 
                    unsafe_allow_html=True
                )

    # Instructions for new users
    if len(st.session_state.conversation_history) == 0:
        st.markdown("---")
        st.info("""
        **How to get started:**
        - 💬 Type your message above and press Enter
        - 🎤 Click 'Speak' to use voice input (speak naturally, pause when done)
        - 🔄 Use 'Retry Voice' if voice system has issues
        
        **Voice Tips:**
        - Speak clearly and at normal pace
        - Wait for the system to stop listening before speaking again
        - You can mix text and voice input as needed
        """)

    # Cleanup on app close
    if st.session_state.voice_system and hasattr(st.session_state.voice_system, 'cleanup'):
        # Register cleanup (though Streamlit doesn't have perfect cleanup hooks)
        pass

if __name__ == "__main__":
    main()