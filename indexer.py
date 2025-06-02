# Imports
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import json
import re

# Set up the embedding model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# File path
input_file = r"G:\Meine Ablage\Documentos\Projects\App-Suite\Religion_App\LLM-Models\Texts\Bibel_new.txt"

# Load verses from .txt file
with open(input_file, 'r', encoding='UTF-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# Regex to extract Book, Chapter, Verse
verse_pattern = re.compile(r"^(.*?) (\d+):(\d+)\s+(.*)$")

chunks = []
chunk_size = 10
temp_chunk = []
verse_map = []

for i, line in enumerate(lines):
    match = verse_pattern.match(line)
    if not match:
        continue

    book, chapter, verse, text = match.groups()
    chapter, verse = int(chapter), int(verse)

    temp_chunk.append(text)
    verse_map.append({
        "verse": verse,
        "text": text
    })

    # Save chunk every N verses or at the end
    is_last_line = i == len(lines) - 1
    if len(temp_chunk) == chunk_size or is_last_line:
        start_verse = verse_map[0]["verse"]
        end_verse = verse_map[-1]["verse"]
        full_text = " ".join(temp_chunk)

        chunk_data = {
            "book": book,
            "chapter": chapter,
            "verses": f"{start_verse}-{end_verse}",
            "verse_map": verse_map.copy(),
            "text": full_text
        }

        chunks.append(chunk_data)
        temp_chunk = []
        verse_map = []

# Convert to JSON strings for embedding
texts = [json.dumps(entry, ensure_ascii=False) for entry in chunks]

# Upload to Qdrant
doc_store = QdrantVectorStore.from_texts(
    texts,
    embeddings,
    url="https://1603c347-3ae3-41e1-9607-40d5d9aacbeb.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.tG4fNFZHtTw2ndjsW0BoFAx-vtP0PJYqnDkP0-p4oSo",
    collection_name="Bible Chunks",
)
