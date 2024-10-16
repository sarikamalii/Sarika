from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# loaidng the embedding model
embeddings = HuggingFaceEmbeddings()

loader = DirectoryLoader(path="data",
                         glob="./*.pdf",
                         loader_cls=UnstructuredFileLoader)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,       
    chunk_overlap=500    
)

# Split the documents
text_chunks = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    persist_directory="vector_db_dir"
)

print("Documents Vectorized")