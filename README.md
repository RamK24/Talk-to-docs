# 📄 Document Query API

This is a FastAPI-based backend that allows users to upload documents (PDFs), process them into searchable formats (semantic & lexical), and query their contents intelligently using a combination of vector embeddings and BM25 search.

---

## ⚙️ Setup

### 🔧 1. Clone the Repository

```bash
git clone https://github.com/your-username/document-query-api.git
cd Talk-to-docs

```
To create index and chunks (This is an offline operation)
```python
python scripts/index_documents.py
```
You can run generate.py locally or use Postman to send queries to http://127.0.0.1:8000/query to query your documents.
```Python
python app/services/generate.py
```

Postman
```bash
http://127.0.0.1:8000/query
Request body = {
  "query": "your query",
}
```
