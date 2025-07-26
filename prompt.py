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

    def generate_system_prompt(self, therapy_type: TherapyType, pdf_context: str = "") -> str:
        prompt = f"""
        You are an experienced AI therapist specializing in {therapy_type.value}. 
        Use the following clinical knowledge extracted from documents to inform your responses when relevant:
        {pdf_context}
        Respond with therapeutic insights and techniques, always keeping the user's wellbeing in focus.
        """
        return prompt.strip()

    def create_conversation_messages(self, user_input: str, pdf_context: str = "", conversation_history: List[Dict] = None) -> List[Dict]:
        therapy_type = self.detect_therapy_type(user_input)
        system_prompt = self.generate_system_prompt(therapy_type, pdf_context)
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_input})
        
        return messages

    def ensure_response_length(self, response: str) -> str:
        return response  
