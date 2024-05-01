import os
import tempfile
import streamlit as st
from embedchain import App

# Configuration of the embedchain app
# Select the LLM and embedding provider as OpenAI, you can choose from cohere,
# anthropic or any other of your choice

# Select the vector database as the opensource chroma db (you are free to choose
# any other vector database of your choice).
def embedchain_bot(db_path, api_key):
    return App.from_config(
        config = {
          "llm" :{"provider": "openai", "config": {"api_key": api_key}},
          "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
          "embedder": {"provider": "openai", "config": {"api_key": api_key}},
        }
    )
# Set up the streamlit app
# Streamlit lets you create user interface with just python code, for this app we will
# . Add a title to the app using 'st.tittle()'
# . Create a text input box for the user to enter their OpenAI API key using 'st.text_input()'
st.title("Chat with PDF")
openai_access_token = st.text_input("OpenAI API Key", type = "password")
# Initiailize the Embedchain App
# . If the OpenAI API key is provided, create a temporary directory for the vector database using 'tempfile.mkdtemp()'
# . Initialize the Embedchain app using the 'embedchain_bot' function
if openai_access_token:
    db_path = tempfile.mkdtemp()
    app = embedchain_bot(db_path, openai_access_token)
    # Upload a PDF file from UI and add it to the knowledge base
    # Use 'st.file_uploader()' to create a file uploader for PDF files.
    # If a PDF file is uploaded, create a temporary file and write the contents of the uploaded fie to it.
    pdf_file = st.file_uploader("Upload a PDF file", type = "pdf")
    # Add the pdf to the knowledge base
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as f:
            f.write(pdf_file.getvalue())
            app.add(f.name, data_type= "pdf_file")
        os.remove(f.name)
        st.success(f"Added {pdf_file.name} to knowledge base!")
    # Ask question about the PDF and display the answer
    # .Create a text input for the user to enter their question using 'st.text_input()'
    # . If a question is asked, get the answer from the Embedchain app and display it using 'st.write()'
    prompt = st.text_input("Ask a questin about the PDF!")
    # Display the answer
    if prompt:
        answer = app.chat(prompt)
        st.write(answer)



