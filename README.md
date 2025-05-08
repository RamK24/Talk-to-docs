# 📄 Document Query API

This is a FastAPI-based backend that allows users to upload documents (PDFs), process them into searchable formats (semantic & lexical), and query their contents intelligently using a combination of vector embeddings and BM25 search.

---

## ⚙️ Setup

### 🔧 1. Clone the Repository


```bash
git clone https://github.com/your-username/document-query-api.git
cd Talk-to-docs

```
## 🧪 Environment Setup (with pip)

Follow these steps to set up the project using `pip` and a virtual environment.

### ✅ 1. Create and activate a virtual environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To create index and chunks (This is an offline operation to be executed mandatorily before querying)
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
