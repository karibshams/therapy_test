from typing import Dict, List
from enum import Enum
import re

class TherapyType(Enum):
    CBT = "Cognitive Behavioral Therapy"
    DBT = "Dialectical Behavior Therapy"
    ACT = "Acceptance and Commitment Therapy"
    GRIEF = "Grief Counseling"
    ANXIETY = "Anxiety Management"
    PARENTING = "Parenting Support"
    DEPRESSION = "Depression Support"
    TRAUMA = "Trauma-Informed Therapy"
    GENERAL = "General Therapy"

class ConversationStyle(Enum):
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    EMPATHETIC = "empathetic"
    MOTIVATIONAL = "motivational"
    GENTLE = "gentle"
    NON_JUDGMENTAL = "non-judgmental"
    SUPPORTIVE = "supportive"
    REFLECTIVE = "reflective"
    EMPOWERING = "empowering"
    CURIOUS = "curious"
    SOLUTION_FOCUSED = "solution-focused"

class InputMode(Enum):
    TEXT = "text"
    VOICE = "voice"
    REAL_TIME_VOICE = "real_time_voice"

class PromptManager:
    def __init__(self, 
                 default_therapy_type: TherapyType = TherapyType.GENERAL,
                 conversation_style: ConversationStyle = ConversationStyle.EMPATHETIC):
        self.default_therapy_type = default_therapy_type
        self.conversation_style = conversation_style

    def detect_therapy_type(self, user_input: str) -> TherapyType:
        text = user_input.lower()
        if any(k in text for k in ["cognitive behavioral therapy", "cbt"]):
            return TherapyType.CBT
        if any(k in text for k in ["dialectical behavior therapy", "dbt"]):
            return TherapyType.DBT
        if any(k in text for k in ["acceptance and commitment therapy", "act"]):
            return TherapyType.ACT
        if any(k in text for k in ["grief", "loss", "bereavement"]):
            return TherapyType.GRIEF
        if any(k in text for k in ["anxiety", "panic", "worried"]):
            return TherapyType.ANXIETY
        if any(k in text for k in ["parent", "child", "kid", "family"]):
            return TherapyType.PARENTING
        if any(k in text for k in ["depress", "sad", "hopeless"]):
            return TherapyType.DEPRESSION
        if any(k in text for k in ["trauma", "trauma-informed"]):
            return TherapyType.TRAUMA
        return self.default_therapy_type

    def detect_input_mode(self, user_input: str) -> InputMode:
        """Detect if input came from voice or text based on content patterns"""
        text = user_input.lower().strip()
        
        # Voice command patterns
        voice_commands = [
            "start voice", "stop voice", "voice status", "record voice",
            "enable voice", "disable voice", "voice mode on", "voice mode off"
        ]
        
        if any(cmd in text for cmd in voice_commands):
            return InputMode.VOICE
        
        # Voice input characteristics (less punctuation, more conversational)
        voice_indicators = 0
        if not text.endswith(('.', '!', '?')):
            voice_indicators += 1
        if len(text.split()) > 5 and text.count(',') == 0:
            voice_indicators += 1
        if any(filler in text for filler in ['um', 'uh', 'like', 'you know']):
            voice_indicators += 2
        
        if voice_indicators >= 2:
            return InputMode.REAL_TIME_VOICE
        
        return InputMode.TEXT

    def generate_system_prompt(self, therapy_type: TherapyType, pdf_context: str = "", 
                             input_mode: InputMode = InputMode.TEXT) -> str:
        base_prompt = f"""
        You are an experienced AI therapist specializing in {therapy_type.value}. 
        Use the following clinical knowledge extracted from documents to inform your responses when relevant:
        {pdf_context}
        """
        
        # Add input mode specific instructions
        if input_mode == InputMode.VOICE or input_mode == InputMode.REAL_TIME_VOICE:
            voice_prompt = """
            
            VOICE INPUT MODE:
            - The user is communicating through voice input, so responses should be more conversational and natural
            - Use shorter sentences and clearer pronunciation-friendly language
            - Acknowledge when you're responding to voice input
            - Be patient with potential transcription errors or incomplete thoughts
            - Encourage the user to speak naturally and take their time
            """
            base_prompt += voice_prompt
        
        therapy_guidance = """
        
        Respond with therapeutic insights and techniques, always keeping the user's wellbeing in focus.
        Maintain professional boundaries while being warm and supportive.
        """
        
        return (base_prompt + therapy_guidance).strip()

    def create_conversation_messages(self, user_input: str, pdf_context: str = "", 
                                   conversation_history: List[Dict] = None, 
                                   input_mode: InputMode = None) -> List[Dict]:
        therapy_type = self.detect_therapy_type(user_input)
        
        # Auto-detect input mode if not provided
        if input_mode is None:
            input_mode = self.detect_input_mode(user_input)
        
        system_prompt = self.generate_system_prompt(therapy_type, pdf_context, input_mode)
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add input mode context to user message if it's voice input
        if input_mode == InputMode.REAL_TIME_VOICE:
            enhanced_message = f"[Voice input] {user_input}"
            messages.append({"role": "user", "content": enhanced_message})
        else:
            messages.append({"role": "user", "content": user_input})
        
        return messages

    def ensure_response_length(self, response: str) -> str:
        return response
    
    def format_response_for_voice(self, response: str, input_mode: InputMode = InputMode.TEXT) -> str:
        """Format response appropriately for voice output"""
        if input_mode == InputMode.VOICE or input_mode == InputMode.REAL_TIME_VOICE:
            # Make response more voice-friendly
            response = response.replace("*", "")  # Remove asterisks
            response = response.replace("_", "")   # Remove underscores
            response = response.replace("  ", " ") # Remove double spaces
            
            # Add voice-friendly transitions
            voice_transitions = [
                ("However,", "But,"),
                ("Furthermore,", "Also,"),
                ("Additionally,", "And,"),
                ("Nevertheless,", "But,"),
                ("Therefore,", "So,")
            ]
            
            for formal, casual in voice_transitions:
                response = response.replace(formal, casual)
        
        return response

    def create_voice_acknowledgment(self, transcript: str) -> str:
        """Create an acknowledgment message for voice input"""
        if len(transcript.strip()) == 0:
            return "I didn't catch that. Could you please try speaking again?"
        
        if len(transcript.split()) < 3:
            return f"I heard '{transcript}'. Could you share a bit more about what you're thinking or feeling?"
        
        return f"I heard you say: '{transcript}'. Let me respond to that."

    def is_voice_command(self, user_input: str) -> bool:
        """Check if input is a voice control command"""
        voice_commands = [
            "start voice", "stop voice", "voice status", "record voice",
            "enable voice", "disable voice", "voice mode", "turn on voice", "turn off voice"
        ]
        return any(cmd in user_input.lower() for cmd in voice_commands)

    def get_voice_help_text(self) -> str:
        """Return help text for voice commands"""
        return """
        Voice Commands:
        • "start voice" - Begin real-time voice input
        • "stop voice" - End real-time voice input  
        • "voice status" - Check if voice is active
        • "record voice" - Single voice recording
        
        Voice Tips:
        • Speak naturally and clearly
        • Pause briefly when you finish speaking
        • The system will automatically respond when you stop talking
        • You can mix voice and text input as needed
        """