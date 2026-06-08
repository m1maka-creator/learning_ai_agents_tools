from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/multilingual-e5-small')

phrases = [
    "как дела",
    "что нового",
    "привет как ты",
    "питон это язык программирования",
    "я люблю есть пиццу",
    "программирование это круто",
]

embeddings = model.encode(phrases)

# print(f"Фраза: '{phrases[0]}'")
# print(f"Вектор (первые 10 чисел): {embeddings[0][:10]}")
# print(f"Размер вектора: {embeddings[0].shape}")

print(embeddings)
