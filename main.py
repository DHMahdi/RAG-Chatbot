import gradio as gr
import requests
from typing import List, Tuple

API_URL_UPLOAD = "http://127.0.0.1:8000/upload"  # Upload endpoint
API_URL_CHAT = "http://127.0.0.1:8000/chat"  # Chat endpoint

def respond(message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Sends user queries to the backend and retrieves responses."""
    payload = {"query": message}  # Removed 'k' as it's no longer required
    try:
        response = requests.post(API_URL_CHAT, json=payload)
        response.raise_for_status()
        assistant_message = response.json().get("response", "No response")  # Adjusted to match backend response key
        history = history + [(message, assistant_message)]  # Append the new conversation turn
        return history
    except requests.exceptions.RequestException as e:
        history = history + [(message, f"Error: {str(e)}")]
        return history


def upload_pdf(file):
    """Uploads a PDF to the FastAPI backend."""
    try:
        with open(file.name, "rb") as f:
            files = {'file': (file.name, f, 'application/pdf')}
            response = requests.post(API_URL_UPLOAD, files=files)
            response.raise_for_status()  # Raise an exception for HTTP error responses
            return response.json().get("message", "No message in response")
    except requests.exceptions.RequestException as e:
        print(f"Error uploading file: {str(e)}")  # Log the error
        return f"Error uploading file: {str(e)}"
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return f"Unexpected error: {str(e)}"

demo = gr.Blocks()

with demo:
    gr.Markdown("## RAG Chatbot")

    chatbot = gr.Chatbot(label="Chat")
    textbox = gr.Textbox(label="Your message")
    btn = gr.Button("Submit")
    btn.click(fn=respond, inputs=[textbox, chatbot], outputs=chatbot)

    with gr.Row():
        pdf_file = gr.File(label="Upload your PDF document")
        upload_button = gr.Button("Upload Document")
        upload_status = gr.Textbox(label="Upload Status", interactive=False)
        upload_button.click(upload_pdf, inputs=pdf_file, outputs=upload_status)

if __name__ == "__main__":
    demo.launch()
