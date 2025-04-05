import logging
import azure.functions as func
import pandas as pd
import fitz  # PyMuPDF for PDF
import openai
import io
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import load_dotenv

# Load environment variables (for local dev)
load_dotenv()

# 🔐 Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
openai.api_version = "2023-05-15"
openai.api_key = os.environ["AZURE_OPENAI_KEY"]

# 🔎 Azure Cognitive Search Configuration
search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ["AZURE_SEARCH_INDEX"],
    credential=AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
)

def get_embedding(text):
    """Generate embedding using Azure OpenAI."""
    response = openai.Embedding.create(
        input=[text],
        engine="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def process_csv(stream, filename):
    """Convert CSV rows to chunks."""
    df = pd.read_csv(stream)
    chunks = []
    for idx, row in df.iterrows():
        text = " | ".join([f"{col}: {str(val)}" for col, val in row.items()])
        chunks.append({
            "id": f"{filename}_row_{idx}",
            "content": text,
            "embedding": get_embedding(text),
            "metadata": {
                "source": filename,
                "type": "csv",
                "row": idx
            }
        })
    return chunks

def process_pdf(stream, filename):
    """Convert PDF pages to chunks."""
    doc = fitz.open(stream=stream, filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if text.strip():  # Avoid blank pages
            chunks.append({
                "id": f"{filename}_page_{page_num}",
                "content": text,
                "embedding": get_embedding(text),
                "metadata": {
                    "source": filename,
                    "type": "pdf",
                    "page": page_num
                }
            })
    return chunks

def main(blob: func.InputStream):
    """Azure Function entry point triggered by Blob upload."""
    filename = blob.name.split("/")[-1]
    extension = filename.split(".")[-1].lower()
    stream = io.BytesIO(blob.read())
    stream.seek(0)

    chunks = []

    if extension == "csv":
        chunks = process_csv(stream, filename)
    elif extension == "pdf":
        chunks = process_pdf(stream, filename)
    else:
        logging.warning(f"❌ Unsupported file type: {extension}")
        return

    if chunks:
        result = search_client.upload_documents(documents=chunks)
        logging.info(f"✅ Uploaded {len(result)} chunks from {filename}")
    else:
        logging.info(f"⚠️ No valid chunks found in {filename}")
