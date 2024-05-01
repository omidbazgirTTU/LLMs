import os
import io
import tempfile
import streamlit as st
from embedchain import App
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

import PyPDF2
from PyPDF2 import PdfReader
from typing_extensions import Concatenate

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
st.title("ChatDoc by Omid Bazgir")
openai_access_token = st.text_input("OpenAI API Key", type = "password")
# Initiailize the Embedchain App
# . If the OpenAI API key is provided, create a temporary directory for the vector database using 'tempfile.mkdtemp()'
# . Initialize the Embedchain app using the 'embedchain_bot' function

if openai_access_token:
    db_path = tempfile.mkdtemp()
    app = embedchain_bot(db_path, openai_access_token)

    os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
    os.environ['LANGCHAIN_API_KEY'] = 'ls__7a1ed57caf664f9b9f665bc0e9c7f6fa'
    os.environ['OPENAI_API_KEY'] = openai_access_token

    # Upload a PDF file from UI and add it to the knowledge base
    # Use 'st.file_uploader()' to create a file uploader for PDF files.
    # If a PDF file is uploaded, create a temporary file and write the contents of the uploaded fie to it.
    pdf_file = st.file_uploader("Upload a PDF file", type = ["pdf"])
    # Add the pdf to the knowledge base
    if pdf_file:
        # read the pdf file with langchain friendly packages
        pdf_content = pdf_file.read()
        # Convert the content to a byte array
        #pdf_bytes = bytearray(pdf_content)
        pdf_bytes = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        # Get the number of pages in the PDF
        raw_text = ''
        for i, page in enumerate(pdf_reader.pages):
            content = page.extract_text()
            if content:
                raw_text +=content
        #############################
        with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as f:
            f.write(pdf_file.getvalue())
            app.add(f.name, data_type= "pdf_file")
        os.remove(f.name)
        st.success(f"Added {pdf_file.name} to knowledge base!")

        # My RAG System
        # Split
        #text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        #splits = text_splitter.split_documents(raw_text)

        text_splitter = CharacterTextSplitter(separator = "\n",chunk_size = 800,chunk_overlap  = 200,length_function = len)
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()
        document_search = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        user_prompt = st.text_input("Ask a questin about the PDF!")
        if user_prompt:
            docs = document_search.similarity_search(user_prompt)
            st.write(chain.run(input_documents=docs, question=user_prompt))
        
    ## -------------------- end RAG -------------------_##
    # Ask question about the PDF and display the answer
    # .Create a text input for the user to enter their question using 'st.text_input()'
    # . If a question is asked, get the answer from the Embedchain app and display it using 'st.write()'
    #prompt = st.text_input("Ask a questin about the PDF!")
    # Display the answer
    #if prompt:
    #    answer = app.chat(prompt)
    #    st.write(answer)



