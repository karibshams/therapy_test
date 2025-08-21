import streamlit as st
import os
import asyncio
from datetime import datetime
from main import EmothriveAI, EmothriveBackendInterface
from voice_input import RealTimeVoiceInput
import time

# Page config
st.set_page_config(
    page_title="Emothrive AI Test", 
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.voice-recording {
    background-color: #ff4b4b;
    color: white;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
    animation: pulse 1.5s infinite;
}

.voice-processing {
    background-color: #ffa500;
    color: white;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}

.voice-ready {
    background-color: #00d4aa;
    color: white;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.6; }
    100% { opacity: 1; }
}

.chat-message {
    padding: 15px;
    margin: 10px 0;
    border-radius: 10px;
    border-left: 4px solid;
}

.user-message {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
}

.assistant-message {
    background-color: #f3e5f5;
    border-left-color: #9c27b0;
}

.voice-message {
    border-left-color: #ff4b4b !important;
    background-color: #fff3e0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'ai_initialized': False,
        'voice_system': None,
        'is_recording': False,
        'current_transcript': '',
        'conversation_history': [],
        'voice_status': 'ready',
        'last_update': 0,
        'pending_voice_message': None  # New flag for voice messages
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Voice callback functions - Fixed for Streamlit
def on_final_transcript(transcript):
    """Called when final transcript is ready - Fixed version"""
    try:
        if transcript.strip():
            # Add user message to session state safely
            if 'conversation_history' not in st.session_state:
                st.session_state.conversation_history = []
                
            st.session_state.conversation_history.append({
                "role": "user",
                "content": transcript,
                "timestamp": datetime.now(),
                "source": "voice"
            })
            
            # Set flag to process message in main thread
            st.session_state.pending_voice_message = transcript
            
        # Reset voice state
        st.session_state.is_recording = False
        st.session_state.current_transcript = ''
        st.session_state.voice_status = 'ready'
        
    except Exception as e:
        print(f"Callback error: {e}")  # Use print instead of st functions

def on_recording_start():
    """Called when recording starts - Simplified"""
    try:
        st.session_state.is_recording = True
        st.session_state.voice_status = 'recording'
    except:
        pass

def on_recording_stop():
    """Called when recording stops - Simplified"""
    try:
        st.session_state.voice_status = 'processing'
    except:
        pass

# Initialize AI system
@st.cache_resource
def init_ai_system():
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        return None, "OpenAI API Key not found. Please set OPENAI_API_KEY in your environment."
    
    try:
        ai_engine = EmothriveAI(openai_api_key=openai_key)
        backend = EmothriveBackendInterface(ai_engine)
        return (ai_engine, backend), None
    except Exception as e:
        return None, f"Failed to initialize AI: {str(e)}"

# Initialize voice system
def init_voice_system():
    if st.session_state.voice_system is None:
        try:
            voice_system = RealTimeVoiceInput(
                silence_threshold=400,
                silence_duration=2.0
            )
            # Simplified callbacks - only use final transcript
            voice_system.set_callbacks(
                on_final_transcript=on_final_transcript,
                on_recording_start=on_recording_start,
                on_recording_stop=on_recording_stop
            )
            st.session_state.voice_system = voice_system
            return True, None
        except Exception as e:
            return False, str(e)
    return True, None

# Process message synchronously
def process_message_sync(message, source='text'):
    try:
        # Get AI system
        ai_data, error = init_ai_system()
        if error:
            st.error(error)
            return
        
        ai_engine, backend = ai_data
        
        # Create event loop for async call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Process message
            request = {'message': message, 'source': source}
            response = loop.run_until_complete(backend.process_message(request))
            
            if response.get('success'):
                ai_response = response['response']['text']
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'content': ai_response,
                    'source': 'ai',
                    'timestamp': datetime.now(),
                    'therapy_type': response['response'].get('therapy_type', 'General')
                })
            else:
                error_msg = f"Error: {response.get('error', 'Unknown error')}"
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'content': error_msg,
                    'source': 'ai',
                    'timestamp': datetime.now()
                })
        finally:
            loop.close()
            
    except Exception as e:
        st.error(f"Processing error: {str(e)}")

# Main app
def main():
    init_session_state()
    
    st.title("🧠 Emothrive AI - Test Interface")
    st.markdown("*Simple testing interface for AI therapy chatbot with voice input*")
    
    # Initialize systems
    ai_data, ai_error = init_ai_system()
    if ai_error:
        st.error(ai_error)
        st.stop()
    
    voice_success, voice_error = init_voice_system()
    
    # Controls section
    st.markdown("## 🎛️ Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Voice controls
        if voice_success:
            if st.session_state.is_recording:
                if st.button("🛑 Stop Recording", type="secondary"):
                    st.session_state.voice_system.stop_recording()
            else:
                if st.button("🎤 Start Recording", type="primary"):
                    st.session_state.voice_system.start_recording()
        else:
            st.button("🎤 Voice Unavailable", disabled=True)
            if voice_error:
                st.caption(f"Error: {voice_error}")
    
    with col2:
        # Voice status
        if st.session_state.voice_status == 'recording':
            st.markdown('<div class="voice-recording">🎤 Recording... Speak now</div>', unsafe_allow_html=True)
        elif st.session_state.voice_status == 'processing':
            st.markdown('<div class="voice-processing">🔄 Processing speech...</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="voice-ready">✅ Voice Ready</div>', unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Reset Chat"):
            st.session_state.conversation_history = []
            st.session_state.current_transcript = ''
            st.rerun()
    
    # Live transcript display
    if st.session_state.current_transcript or st.session_state.is_recording:
        st.markdown("### 🎤 Live Transcript")
        if st.session_state.current_transcript:
            st.info(f"**Current Speech:** {st.session_state.current_transcript}")
        elif st.session_state.is_recording:
            st.info("**Listening...** Start speaking...")
    
    # Text input section
    st.markdown("## ✏️ Text Input")
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Type your message:",
            placeholder="How are you feeling today?",
            disabled=st.session_state.is_recording
        )
        submitted = st.form_submit_button("Send", disabled=st.session_state.is_recording)
        
        if submitted and user_input.strip():
            # Add user message
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'source': 'text',
                'timestamp': datetime.now()
            })
            
            # Process with AI
            with st.spinner("🤔 AI is thinking..."):
                process_message_sync(user_input, 'text')
            
            st.rerun()
    
    # Conversation display
    st.markdown("## 💬 Conversation")
    
    if not st.session_state.conversation_history:
        st.info("👋 Start a conversation by typing a message or using voice input above!")
        st.markdown("**Suggestions:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("😟 I'm feeling anxious"):
                st.session_state.conversation_history.append({
                    'role': 'user', 'content': "I'm feeling anxious",
                    'source': 'text', 'timestamp': datetime.now()
                })
                process_message_sync("I'm feeling anxious", 'text')
                st.rerun()
        with col2:
            if st.button("💭 I need someone to talk to"):
                st.session_state.conversation_history.append({
                    'role': 'user', 'content': "I need someone to talk to",
                    'source': 'text', 'timestamp': datetime.now()
                })
                process_message_sync("I need someone to talk to", 'text')
                st.rerun()
    else:
        # Display conversation history
        for i, message in enumerate(st.session_state.conversation_history):
            timestamp = message.get('timestamp', datetime.now()).strftime("%H:%M:%S")
            source = message.get('source', 'text')
            
            if message['role'] == 'user':
                source_icon = "🎤" if source == 'voice' else "💬"
                css_class = "chat-message user-message voice-message" if source == 'voice' else "chat-message user-message"
                
                st.markdown(f"""
                <div class="{css_class}">
                    <strong>{source_icon} You ({timestamp}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
            else:  # assistant
                therapy_type = message.get('therapy_type', '')
                therapy_info = f" | {therapy_type}" if therapy_type else ""
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🧠 AI Therapist ({timestamp}{therapy_info}):</strong><br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Debug info (collapsible)
    with st.expander("🔧 Debug Info"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Session State:**")
            st.json({
                'ai_initialized': st.session_state.ai_initialized,
                'is_recording': st.session_state.is_recording,
                'voice_status': st.session_state.voice_status,
                'messages_count': len(st.session_state.conversation_history),
                'current_transcript_length': len(st.session_state.current_transcript)
            })
        
        with col2:
            st.write("**Voice System:**")
            if st.session_state.voice_system:
                status = st.session_state.voice_system.get_status()
                st.json(status)
            else:
                st.write("Voice system not initialized")
    
    # Auto-refresh for voice updates
    if st.session_state.is_recording:
        time.sleep(0.3)
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        st.write("Application stopped")
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)