"""
@author: Ranjit S.
"""

import os
import time
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings

import chromadb
from chromadb.utils import embedding_functions
import openai



# Load environment variables
load_dotenv()

# Global variables
llm = None
embedding_fn = None
OPENAI_API_KEY = None
collection = None

def initialized():
    global llm
    global OPENAI_API_KEY
    
    print("... INVOKE Initialization .....")
    # Load environment variables
    load_dotenv()
      
    # Secure API Key Setup
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","THIS IS UNSET")
    print(OPENAI_API_KEY)

    openai.api_key = OPENAI_API_KEY
   
    # Initialize LLM
    llm = OpenAI(temperature=0.9, max_tokens=500)
    print("LLM Initialized:", llm)

def initialized_db():
    global embedding_fn
    global collection
    
    db_directory = "./db"

    # Initialize the ChromaDB client
    client = chromadb.PersistentClient(db_directory)

    # Collection name
    collection_name = "documents"
     
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
    
    # Check if the collection exists, if not, create it
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        print(f"Collection '{collection_name}' not found. Creating it...")
        collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)
        print(f"Collection '{collection_name}' created with embedding function.")

    # Add some data to the collection
    documents = [
        {"id": "doc1", "content": "This is the first document.", "metadata": {"author": "Ranjit"}},
        {"id": "doc2", "content": "This is the second document.", "metadata": {"author": "S"}}
    ]

    for doc in documents:
        collection.add(
            ids=[doc["id"]],
            documents=[doc["content"]],
            metadatas=[doc["metadata"]],
        )

    print(f"Documents added to collection '{collection_name}'.")
    list_collections()

def list_collections():
    db_directory = "./db"

    # Initialize the ChromaDB client
    client = chromadb.PersistentClient(db_directory)
    collections = client.list_collections()
    print("Collections in the database:")
    for col in collections:
        print(f"- {col.name}")



def docLoadAndSplit(urls):
    print("Loading and splitting documents...")
    
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    print("Data Loading...Started...✅✅✅")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    print("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print("Split documents:", docs)

    # Add 'source' metadata to each document if missing
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = "unknown"
    
    print("Split documents with 'source' metadata:", docs)
    
    # Create embeddings and store in ChromaDB
    vectorEmedding(docs)
    
def vectorEmedding(docs):
    global collection
    print("Creating embeddings and storing in ChromaDB...")
    
    db_directory = "./db"

    # Initialize ChromaDB
    client = chromadb.PersistentClient(db_directory)
   
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
    
    # Get or create the collection
    collection = client.get_or_create_collection(name="documents", embedding_function=embedding_fn)
    print("Collection:", collection)

    # Create embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings()
    print("Embeddings model:", embeddings)

    for doc in docs:
        try:
            # Generate embedding for the document
            embedding_vector = embeddings.embed_query(doc.page_content)
            print("Generated embedding vector for document.")
            
            # Add document to collection
            collection.add(
                documents=[doc.page_content],
                metadatas=[{"source": doc.metadata.get("source", "unknown")}],
                embeddings=[embedding_vector],
                ids=[str(hash(doc.page_content))]
            )
        except openai.RateLimitError as e:
            print("Rate limit exceeded. Please check your OpenAI quota or try again later.")
            print(f"Rate limit error: {e}")
        except openai.OpenAIError as e:
            print(f"An error occurred: {str(e)}")
           
    print("Data Persisted Successfully in ChromaDB ✅✅✅")

def questions(query):
    print("Processing query...")
       
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
    
    # Get the collection
    collection = get_db()
    
    # Check if the query is already 
    cache_start_time = time.time()
    cached_result = get_cached_result(query, embedding_fn, collection)
    if cached_result:
        print("")
        print("=========================================================")
        print("Cache hit! Returning cached result.")
        cache_end_time = time.time()
        cache_lookup_time = cache_end_time - cache_start_time
        print(f"Cache lookup time: {cache_lookup_time:.2f} seconds")
        print("===========================================================")
        print("")
        print("cached_result---->:", cached_result)
        print("")
        return cached_result

    # If not cached, process the query and cache the result
    result = expensive_processing_function(query , embedding_fn, collection)
    #(query, result, embedding_fn, collection)
    return result

def expensive_processing_function(query, embedding_fn, collection, k=3):
    print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Running enhanced RAG logic...")

    rag_start_time = time.time()

    # Step 1: Embed the query
    query_embedding = embedding_fn([query])[0]

    # Step 2: Similarity search (top-k)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    chunks = results["documents"][0]
    sources = results["metadatas"][0]

    print(f"Top {k} chunks retrieved for query.")

    # Step 3: Construct the prompt
    context = "\n\n".join(chunks)
    prompt = f"""Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

    # Step 4: Query LLM
    llm = OpenAI(temperature=0.7, max_tokens=500)
    answer = llm(prompt)

    # Step 5: Cache & return
    cache_query_result(query, answer, embedding_fn, collection)

    rag_end_time = time.time()
    print(f"RAG execution time: {rag_end_time - rag_start_time:.2f} seconds")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    return answer
    
def expensive_processing_function_org(query , embedding_fn, collection):
    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")    
    print("Running expensive processing function...")
    rag_start_time = time.time()
    
    db_directory = "./db"
    
    # Initialize LLM
    llm = OpenAI(temperature=0.9, max_tokens=500)
    
    # Load ChromaDB and set up retriever
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(collection_name="documents", embedding_function=embeddings, persist_directory=db_directory)

    # Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever()

    # Set up the retrieval chain
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    result = chain({"question": query}, return_only_outputs=True)

    # Display the answer
    print("Answer:", result["answer"])
    
    # Cache the result before returning it
    cache_query_result(query, result["answer"], embedding_fn, collection)
    
    rag_end_time = time.time()
    rag_execution_time = rag_end_time - rag_start_time
    print(f"RAG execution time: {rag_execution_time:.2f} seconds")
    print("")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")   
    return result["answer"]

def cache_query_result(query, result, embedding_fn, collection):
    print("Caching query result...")
    
    # Generate embedding for the query
    query_embedding = embedding_fn([query])[0]

    # Store the query, its embedding, and the result in ChromaDB
    collection.add(
        documents=[query],
        embeddings=[query_embedding],
        metadatas=[{"result": result, "source": "cached_query"}],
        ids=[str(hash(query))]
    )

def get_cached_result(query, embedding_fn, collection, threshold=0.8):
    print("Checking cache for query...")
    
    # Generate embedding for the query
    query_embedding = embedding_fn([query])[0]

    # Query the collection for similar embeddings
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1  # Return the top match
    )

    # Check if the similarity score meets the threshold
    if results["distances"][0][0] <= (1 - threshold):
        metadata = results["metadatas"][0][0]
        if "result" in metadata:  # Check if "result" key exists
            return results["metadatas"][0][0]["result"]  # Return the cached result
            
    return None  # No cache hit or missing "result" key

def get_db():
    global collection
    
    db_directory = "./db"

    # Initialize the ChromaDB client
    client = chromadb.PersistentClient(db_directory)

    # Collection name
    collection_name = "documents"
      
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)
    
    # Check if the collection exists, if not, create it
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except:
        print(f"Collection '{collection_name}' not found. Creating it...")
        collection = client.create_collection(name=collection_name, embedding_function=embedding_fn)
        
    return collection
