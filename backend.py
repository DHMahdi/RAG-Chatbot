# Importing necessary libraries
import os  
import tempfile 
import shutil
import logging 
from typing import List 
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel 
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import JSONResponse 

# Global variables
faiss_index = None  # Holds the FAISS index for semantic search
chunk_documents = None  # Stores the document chunks used for searching
UPLOAD_DIR = "uploaded_files"  # Directory to save uploaded files

# Initialize the embedding model for semantic search (SBERT)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

# Initialize the Flan-T5 text generation model
text_gen_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
text_gen_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)

# Ensure the upload directory exists (creates it if not present)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define the structure for chat requests using Pydantic
class ChatRequest(BaseModel):
    query: str  # Input question/query from the user

# Function to generate a response using Flan-T5
def generate_response(query: str, context: str) -> str:
    """
    Generate a text response based on the query and context using the Flan-T5 model.

    Args:
        query (str): The user-provided question.
        context (str): Relevant text context from the uploaded document.

    Returns:
        str: The generated response from the Flan-T5 model.
    """
    input_text = f"question: {query} context: {context}"  # Input format expected by the model
    inputs = text_gen_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)  # Tokenize and prepare input
    outputs = text_gen_model.generate(inputs.input_ids, max_length=200, num_beams=3, early_stopping=True)  # Generate response
    return text_gen_tokenizer.decode(outputs[0], skip_special_tokens=True)  # Decode and clean output

# Initialize the FastAPI application
app = FastAPI()

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/upload")
async def upload_file(file: UploadFile):
    """
    Endpoint to upload and process PDF files.

    Args:
        file (UploadFile): The uploaded file object.

    Returns:
        JSONResponse: Status message or error details.
    """
    try:
        logger.info(f"Received file: {file.filename}")

        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise ValueError("Only PDF files are allowed")

        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_file_path = tmp.name

        # Move the temporary file to the permanent upload directory
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        shutil.move(tmp_file_path, file_path)
        logger.info(f"File saved at: {file_path}")

        # Verify PDF integrity
        try:
            with open(file_path, "rb") as pdf_file:
                reader = PdfReader(pdf_file)
                if reader.is_encrypted:  # Decrypt if the file is encrypted
                    reader.decrypt('')
                _ = len(reader.pages)  # Check that the PDF has pages
        except Exception as e:
            raise ValueError(f"Invalid PDF file: {str(e)}")

        # Load and process the PDF
        loader = PyPDFLoader(file_path)  # Load the file as LangChain documents
        documents = loader.load()  # Extract documents from the PDF
        logger.info(f"Loaded {len(documents)} document(s) from PDF")

        # Split documents into smaller chunks for efficient processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        global chunk_documents
        chunk_documents = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(chunk_documents)} chunks")

        # Create a FAISS index from the chunks for semantic search
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
    """
    Endpoint to query the uploaded document using semantic search and text generation.

    Args:
        req (ChatRequest): The user's query as a ChatRequest object.

    Returns:
        dict: The user's query and the generated response.
    """
    try:
        # Ensure an index has been created from uploaded files
        if not faiss_index:
            raise ValueError("No document processed. Please upload a file first.")

        # Perform a semantic search for the most relevant document chunk
        search_results = faiss_index.similarity_search(req.query, k=1)  # Retrieve top result

        # Handle no results found
        if not search_results:
            return {"query": req.query, "response": "No relevant information found in the uploaded documents."}

        # Combine the most relevant chunk(s) as context
        relevant_text = " ".join([result.page_content for result in search_results])

        # Generate a response based on the query and context
        response = generate_response(req.query, relevant_text)

        return {"query": req.query, "response": response}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return {"error": str(e)}
