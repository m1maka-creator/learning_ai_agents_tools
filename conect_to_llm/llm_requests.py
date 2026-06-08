import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Объясни что такое RAG в ИИ-агентах простыми словами",
        }
    ],
    model="llama-3.1-8b-instant",
)

print(chat_completion.choices[0].message.content)

print(json.dumps(chat_completion.model_dump(), indent=2, ensure_ascii=False))