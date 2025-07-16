from typing import Dict, List
from enum import Enum

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

class PromptManager:
    def __init__(self, 
                 default_therapy_type: TherapyType = TherapyType.GENERAL,
                 conversation_style: ConversationStyle = ConversationStyle.EMPATHETIC):
        self.default_therapy_type = default_therapy_type
        self.conversation_style = conversation_style

    def detect_therapy_type(self, user_input: str) -> TherapyType:
        text = user_input.lower()
        if any(k in text for k in ["grief", "loss", "bereavement"]):
            return TherapyType.GRIEF
        if any(k in text for k in ["anxiety", "panic", "worried"]):
            return TherapyType.ANXIETY
        if any(k in text for k in ["parent", "child", "kid", "family"]):
            return TherapyType.PARENTING
        if any(k in text for k in ["depress", "sad", "hopeless"]):
            return TherapyType.DEPRESSION
        return self.default_therapy_type

    def generate_system_prompt(self, 
                               therapy_type: TherapyType,
                               pdf_context: str = "") -> str:
        prompt = f"""
You are an experienced AI therapist specializing in {therapy_type.value}. 

Your approach combines Cognitive Behavioral Therapy (CBT), Dialectical Behavior Therapy (DBT), and Acceptance and Commitment Therapy (ACT). 
Respond in a warm, compassionate, and professional therapist tone. 

Use the following clinical knowledge extracted from documents to inform your responses when relevant:
{pdf_context}

Provide a detailed, session-style reply resembling a live therapy session — 
engage with the user’s feelings, thoughts, and behaviors; 
offer insights and therapeutic techniques; 
use examples and ask reflective questions as appropriate.

Be supportive, patient, and focus on practical coping strategies and problem solving.
"""
        return prompt.strip()

    def create_conversation_messages(self, 
                                     user_input: str,
                                     pdf_context: str = "",
                                     conversation_history: List[Dict] = None) -> List[Dict]:
        therapy_type = self.detect_therapy_type(user_input)
        system_prompt = self.generate_system_prompt(therapy_type, pdf_context)
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_input})
        
        return messages
