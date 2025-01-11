# Import necessary libraries.
import gradio as gr  
import requests  
from typing import List, Tuple  

# API endpoints for backend interaction
API_URL_UPLOAD = "http://127.0.0.1:8000/upload"  # Endpoint for uploading PDF documents
API_URL_CHAT = "http://127.0.0.1:8000/chat"  # Endpoint for querying the chatbot

def respond(message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Handles sending user messages to the chatbot backend and retrieving responses.

    Args:
        message (str): The user's input message.
        history (List[Tuple[str, str]]): Chat history as a list of (user_message, assistant_response) tuples.

    Returns:
        List[Tuple[str, str]]: Updated chat history including the latest conversation turn.
    """
    payload = {"query": message}  # Prepare the payload with the user's query
    try:
        # Make a POST request to the chatbot endpoint with the user's query
        response = requests.post(API_URL_CHAT, json=payload)
        response.raise_for_status()  # Raise an error for HTTP response codes >= 400
        assistant_message = response.json().get("response", "No response")  # Extract the assistant's response
        # Append the new user-assistant exchange to the chat history
        history = history + [(message, assistant_message)]
        return history
    except requests.exceptions.RequestException as e:
        # Handle exceptions related to HTTP requests (e.g., connection errors, timeouts)
        history = history + [(message, f"Error: {str(e)}")]  # Log the error in chat history
        return history

def upload_pdf(file):
    """
    Handles uploading a PDF document to the FastAPI backend for processing.

    Args:
        file: The file object of the PDF to be uploaded.

    Returns:
        str: A success message from the backend or an error description.
    """
    try:
        # Open the file in binary mode and send it to the backend as a multipart/form-data request
        with open(file.name, "rb") as f:
            files = {'file': (file.name, f, 'application/pdf')}  # Prepare the file payload
            response = requests.post(API_URL_UPLOAD, files=files)  # POST request to upload endpoint
            response.raise_for_status()  # Raise an exception for HTTP error responses
            # Return the success message from the backend
            return response.json().get("message", "No message in response")
    except requests.exceptions.RequestException as e:
        # Handle HTTP-related errors
        print(f"Error uploading file: {str(e)}")  # Log the error
        return f"Error uploading file: {str(e)}"
    except Exception as e:
        # Handle unexpected errors (e.g., file access issues)
        print(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}"

# Define the Gradio interface
demo = gr.Blocks()  # Initialize a Gradio Blocks container for building the UI

with demo:
    # Add a title to the Gradio interface
    gr.Markdown("## RAG Chatbot")  # Markdown for styling the title

    # Create a chatbot interface
    chatbot = gr.Chatbot(label="Chat")  # Chatbot widget for displaying conversations
    textbox = gr.Textbox(label="Your message")  # Input box for user messages
    btn = gr.Button("Submit")  # Submit button to send messages
    # Link the button click to the `respond` function
    btn.click(fn=respond, inputs=[textbox, chatbot], outputs=chatbot)

    # Create a file upload section
    with gr.Row():  # Arrange widgets in a horizontal row
        pdf_file = gr.File(label="Upload your PDF document")  # File upload widget for PDF documents
        upload_button = gr.Button("Upload Document")  # Button to trigger the upload
        upload_status = gr.Textbox(label="Upload Status", interactive=False)  # Read-only textbox for upload status
        # Link the upload button click to the `upload_pdf` function
        upload_button.click(upload_pdf, inputs=pdf_file, outputs=upload_status)

# Launch the Gradio interface
if __name__ == "__main__":
    demo.launch()  # Starts a local server and opens the UI in a browser
