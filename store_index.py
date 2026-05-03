from src.helper import load_pdf_files, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents

extracted_data=load_pdf_files("data")


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )

        )
    return minimal_docs
minimal_docs=filter_to_minimal_docs(extracted_data)

def text_split(minimal_docs):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk= text_splitter.split_documents(minimal_docs)
    return texts_chunk

texts_chunk=text_split(minimal_docs)

def download_embeddings():
    model_name= "sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceBgeEmbeddings(
        model_name=model_name
    )
    return embeddings
embedding=download_embeddings()

vector= embedding.embed_query("hello world")
print(len(vector))

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"]= PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY


pinecone_api_key= PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key)
pc


index_name="medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index=pc.Index(index_name)

docsearch=PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)
