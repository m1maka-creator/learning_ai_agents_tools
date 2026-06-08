import os
import json
from groq import Groq
from dotenv import load_dotenv


load_dotenv() 


client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

messages = [
    {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко и по делу."}
]

print("Чат запущен. Введи 'quit' чтобы выйти.\n")

r = input("Ты: ")

while r != "quit":

    messages.append({"role": "user", "content": r})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=512
    )

    reply = response.choices[0].message.content

    messages.append({"role": "assistant", "content": reply})

    print(f"\nБот: {reply}")
    print(f"[сообщений в истории: {len(messages)}]\n")

    r = input("Ты: ")

print("Чат завершён.")