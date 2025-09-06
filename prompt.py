from typing import Dict, List, Optional
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

    def detect_user_mood(self, user_input: str) -> Optional[str]:
        """Detect user's emotional state for voice adaptation"""
        text = user_input.lower()
        
        mood_indicators = {
            'sad': ['sad', 'depressed', 'down', 'hopeless', 'cry', 'crying'],
            'anxious': ['anxious', 'worried', 'nervous', 'panic', 'scared', 'afraid'],
            'angry': ['angry', 'mad', 'frustrated', 'furious', 'annoyed'],
            'happy': ['happy', 'good', 'great', 'wonderful', 'excited', 'joy']
        }
        
        for mood, indicators in mood_indicators.items():
            if any(indicator in text for indicator in indicators):
                return mood
        
        return None

   

    def generate_system_prompt(self, therapy_type: TherapyType, pdf_context: str = "", is_voice_input: bool = False) -> str:
        base_prompt = f"""
        You are an experienced AI therapist specializing in {therapy_type.value}. 
        Use the following clinical knowledge extracted from documents to inform your responses when relevant:
        {pdf_context}
        Respond with therapeutic insights and techniques, always keeping the user's wellbeing in focus.
        """
        
        if is_voice_input:
            voice_addition = """
        
        VOICE INTERACTION GUIDELINES:
        - The user is speaking to you directly, so be extra warm and conversational
        - Keep responses natural and flowing for spoken conversation
        - Use shorter sentences that are easier to understand when heard
        - Add natural conversational markers like "I hear you" or "That makes sense"
        - Avoid complex terminology that might be hard to follow in audio
        - Be more empathetic and personal in your tone since voice feels more intimate
        """
            base_prompt += voice_addition
        
        return base_prompt.strip()

    def create_conversation_messages(self, 
                                   user_input: str, 
                                   pdf_context: str = "", 
                                   conversation_history: List[Dict] = None,
                                   is_voice_input: bool = False) -> List[Dict]:
        therapy_type = self.detect_therapy_type(user_input)
        system_prompt = self.generate_system_prompt(therapy_type, pdf_context, is_voice_input)
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_input})
        
        return messages

    def get_voice_emotion_for_response(self, response_text: str) -> str:
        """Determine appropriate voice emotion based on response content"""
        text_lower = response_text.lower()
        
        if any(word in text_lower for word in ['sorry', 'difficult', 'hard', 'challenging']):
            return 'empathetic'
        elif any(word in text_lower for word in ['great', 'wonderful', 'proud', 'progress']):
            return 'friendly'
        elif any(word in text_lower for word in ['calm', 'relax', 'gentle', 'peaceful']):
            return 'gentle'
        else:
            return 'empathetic'

    def ensure_response_length(self, response: str) -> str:
        return response