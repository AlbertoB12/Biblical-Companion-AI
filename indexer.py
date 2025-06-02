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
input_file = r""

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
    url="",
    api_key="",
    collection_name="Bible Chunks",
)
