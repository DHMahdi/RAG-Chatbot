from fastapi import FastAPI, UploadFile
from typing import List
from pydantic import BaseModel
import os
import tempfile
import shutil
import logging
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import JSONResponse

# Global variables for the FAISS index and chunked documents
faiss_index = None
chunk_documents = None
UPLOAD_DIR = "uploaded_files"

# Initialize the embedding model (SBERT)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Initialize the text generation model (Flan-T5-base)
text_gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
text_gen_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)

# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Pydantic model for chat requests
class ChatRequest(BaseModel):
    query: str

# Generate response using Flan-T5
def generate_response(query: str, context: str) -> str:
    input_text = f"question: {query} context: {context}"
    inputs = text_gen_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = text_gen_model.generate(inputs.input_ids, max_length=200, num_beams=3, early_stopping=True)
    return text_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        logger.info(f"Received file: {file.filename}")

        if not file.filename.endswith(".pdf"):
            raise ValueError("Only PDF files are allowed")

        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_file_path = tmp.name

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        shutil.move(tmp_file_path, file_path)
        logger.info(f"File saved at: {file_path}")

        # Verify PDF integrity
        try:
            with open(file_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                if reader.is_encrypted:
                    reader.decrypt('')
                _ = len(reader.pages)
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {str(e)}")

        # Process the PDF and chunk text
        loader = PyPDFLoader(file_path)
        documents = loader.load()  # Returns a list of `Document` objects
        logger.info(f"Loaded {len(documents)} document(s) from PDF")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        global chunk_documents
        chunk_documents = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(chunk_documents)} chunks")

        # Embed chunks and create FAISS index
        global faiss_index
        faiss_index = FAISS.from_documents(chunk_documents, embedding_model)
        logger.info("FAISS index created")

        return {"message": f"File '{file.filename}' uploaded and processed successfully!"}

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        return JSONResponse(content={"detail": str(ve)}, status_code=400)
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return JSONResponse(content={"detail": f"Error uploading file: {str(e)}"}, status_code=500)

@app.post("/chat")
async def chat(req: ChatRequest):
    """Endpoint to query the processed document."""
    try:
        if not faiss_index:
            raise ValueError("No document processed. Please upload a file first.")

        # Perform semantic search to find the most relevant document chunk
        search_results = faiss_index.similarity_search(req.query, k=1)  # Retrieve the top result

        if not search_results:
            return {"query": req.query, "response": "No relevant information found in the uploaded documents."}

        # Concatenate the relevant chunk
        relevant_text = " ".join([result.page_content for result in search_results])

        # Generate a response using Flan-T5
        response = generate_response(req.query, relevant_text)

        return {"query": req.query, "response": response}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}
