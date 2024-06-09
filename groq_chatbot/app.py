import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain import hub
import pickle

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Directory to save uploaded PDFs and vector stores
UPLOAD_DIRECTORY = "uploaded_pdfs"
VECTOR_STORE_DIRECTORY = "vector_stores"

# Ensure the directories exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
if not os.path.exists(VECTOR_STORE_DIRECTORY):
    os.makedirs(VECTOR_STORE_DIRECTORY)

index = 0  # Initialize an index variable outside the main function

def main():
    global index  # Access the global index variable
    st.set_page_config(page_title="PDF QA Chatbot", page_icon="🤖", layout="wide")
    st.title("PDF QA Chatbot")
   

    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [groq](https://console.groq.com/playground) LLM model
        
        ## Instructions
        1. **Upload a PDF**: Use the file uploader to select and upload your PDF document.
        2. **Ask Questions**: Once the PDF is processed, type your questions related to the PDF content in the provided text box.
        3. **Get Answers**: Click the "Get Answer" button to receive responses from the chatbot based on the content of the uploaded PDF.
        ''')

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf is not None:
        with st.spinner("Processing PDF..."):
            # Save the uploaded file to the upload directory
            file_path = os.path.join(UPLOAD_DIRECTORY, pdf.name)
            with open(file_path, "wb") as f:
                f.write(pdf.read())

            # Generate vector store file path
            vector_store_path = os.path.join(VECTOR_STORE_DIRECTORY, f"{pdf.name[:-4]}.pkl")
            
            # Check if the vector store is already loaded in the session state
            if 'vector_store' not in st.session_state or st.session_state['vector_store_name'] != pdf.name:
                # Check if the vector store file exists
                if os.path.exists(vector_store_path):
                    with open(vector_store_path, "rb") as f:
                        st.session_state['vector_store'] = pickle.load(f)
                    st.session_state['vector_store_name'] = pdf.name
                    st.success('Vector store loaded from disk.')
                else:
                    # Load the PDF using the file path
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=20,
                    )
                    documents = text_splitter.split_documents(pages)
                    db = FAISS.from_documents(documents, OllamaEmbeddings())
                    
                    # Save the vector store to the session state and disk
                    st.session_state['vector_store'] = db
                    st.session_state['vector_store_name'] = pdf.name
                    with open(vector_store_path, "wb") as f:
                        pickle.dump(db, f)
                    st.success('Vector store created and saved to disk.')

        llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    
        prompt = ChatPromptTemplate.from_messages([
                    ("human", """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
                    If you don't know the answer, just say that you don't know. Use ten sentences maximum and keep the answer concise
                    Question: {input} 
                    Context: {context} 
                    Answer:"""),
                    ])  # Adjust this as per your actual prompt retrieval method
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state['vector_store'].as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        
        st.success("PDF processed successfully. You can now ask questions about the content.")
        
        question = st.text_input(f"Ask a question about the PDF content")

        if question:
            with st.spinner("Getting the answer..."):
                response = retriever_chain.invoke({"input": question})
                st.success(response["answer"])
                #st.write(response)

if __name__ == "__main__":
    main()
