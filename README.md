
````markdown
# 🤖 Loan Approval RAG Q&A Chatbot

A lightweight Retrieval-Augmented Generation (RAG) chatbot that intelligently answers natural language questions about loan approval — powered by OpenAI GPT-3.5 and FAISS-based document retrieval.

---

## 📊 Dataset

- **Source**: [Kaggle - Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)
- **Filename**: `Training Dataset.csv`

---

## 🚀 Features

- 📄 Converts CSV loan records into semantic documents
- 🧠 Embeds each document using `sentence-transformers`
- 🔎 Retrieves top relevant entries with FAISS
- 💬 Sends context + query to OpenAI GPT-3.5
- ✅ Works with free OpenAI credits (via API)
- ☁️ Fully compatible with Google Colab or local environment

---

## 📦 Installation

Install required libraries locally or in Colab:

```bash
pip install openai==1.2.4 faiss-cpu sentence-transformers pandas numpy
````

---

## 🔐 API Key Setup

Set your OpenAI API Key securely in code:

```python
import openai
openai.api_key = "your-openai-api-key"
```

Or with environment variable:

```python
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

---

## 🧠 How It Works

1. Loads CSV as tabular documents
2. Uses `all-MiniLM-L6-v2` to convert rows into vector embeddings
3. Builds FAISS index to enable similarity search
4. Retrieves top-k most relevant rows per user query
5. Uses OpenAI's GPT-3.5-turbo to generate human-like, grounded answers

---

## ✅ Usage in Google Colab

1. Upload the dataset (`Training Dataset.csv`)
2. Paste in the code
3. Add your OpenAI key
4. Ask questions interactively

---

## 💡 Sample Questions

```text
Why might a loan be rejected?
Does credit history impact approval?
Is gender a factor in loan decisions?
What is the effect of property area?
```

---

## 🧪 Example Code (Colab-compatible)

```python
# Install libraries (Colab)
!pip install openai==1.2.4 faiss-cpu sentence-transformers pandas numpy --quiet

# Imports
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

# Load CSV (after uploading to Colab)
df = pd.read_csv("Training Dataset.csv")

# Turn each row into a semantic document
docs = []
for _, row in df.iterrows():
    text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
    docs.append(text)

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(docs, convert_to_numpy=True)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Init OpenAI client
client = OpenAI(api_key="your-api-key-here")  # Replace with your key

# Retrieval + Generation
def retrieve(query, top_k=5):
    query_vec = model.encode([query])[0]
    _, I = index.search(np.array([query_vec]), top_k)
    return [docs[i] for i in I[0]]

def generate_answer(context, query):
    prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about loan approval."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

# Ask question
query = "Does credit history matter?"
context = "\n\n".join(retrieve(query))
print("Answer:\n", generate_answer(context, query))
```

---

## 📁 Project Structure

```
loan-rag-chatbot/
├── Training Dataset.csv
├── rag_chatbot.ipynb         # (Optional: Notebook version)
├── README.md
```

---

## 📬 Future Enhancements

* Gradio or Streamlit UI
* Chat memory for multi-turn Q\&A
* Support Claude/Gemini fallback
* Fine-tuned loan domain model

---

## ⚖️ License

MIT License — for educational use. Dataset © respective Kaggle uploader.

---

## 🙏 Credits

* [Sonalisingh1411 - Kaggle Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction)
* [OpenAI](https://platform.openai.com/)
* [Sentence-Transformers](https://www.sbert.net/)
* [FAISS by Facebook Research](https://github.com/facebookresearch/faiss)

```

---

