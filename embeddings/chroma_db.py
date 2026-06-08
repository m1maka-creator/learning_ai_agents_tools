import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="my_notes")

model = SentenceTransformer("intfloat/multilingual-e5-small")
documents = [
    "Gradient boosting строит деревья последовательно, каждое исправляет ошибки предыдущего",
    "Random Forest обучает деревья параллельно на случайных подвыборках",
    "Нейронная сеть — это граф из слоёв с весами, обучается через backprop",
    "PCA снижает размерность, сохраняя максимум дисперсии",
    "K-means кластеризует точки вокруг k центроидов",
]

ids = [f"doc_{i}" for i in range(len(documents))]
metadatas = [{"topic": "ML", "index": i} for i in range(len(documents))]

embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

query = "как работает ансамблевый метод с деревьями"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={"topic": "ML"}  
)


for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"[{distance:.3f}] {doc}")