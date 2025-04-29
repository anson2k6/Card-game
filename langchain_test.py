from pathlib import Path
import pickle

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def process_pdf(pdf_path, embedding_path=None):
    print(f"Processing {pdf_path}...")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    vectorstore = FAISS.from_documents(chunks, embeddings)

    save_path = embedding_path or f"{Path(pdf_path).stem}_vectorstore.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(vectorstore, f)
    print(f"Saved vectorstore to {save_path}")

    return vectorstore

def load_vectorstore(file_path):
    with open(file_path, 'rb') as f:
        vectorstore = pickle.load(f)
    return vectorstore

def create_qa_chain(vectorstore):
    retriever = vectorstore.as_retriever()
    llm = Ollama(model='gemma3:4b')
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def query_pdf(qa_chain):
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        answer = qa_chain.run(query)
        print(f"\nAnswer: {answer}")

if __name__ == "__main__":
    pdf_path = "ktane_manual.pdf"
    vectorstore = process_pdf(pdf_path)


    qa_chain = create_qa_chain(vectorstore)

    query_pdf(qa_chain)
