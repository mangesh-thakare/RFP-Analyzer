from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Optional
from datetime import datetime
import os
import sys
import warnings

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_extraction import extract_text_from_document, preprocess_text

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class DocumentStore:
    def __init__(self, persist_directory: str = "./vector_db"):
        """
        Initialize the document store
        Args:
            persist_directory: Directory to store the vector database
        """
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", "? ", "! ", ", "]
            )
            
            self.persist_directory = persist_directory
            os.makedirs(persist_directory, exist_ok=True)
            
            if os.path.exists(persist_directory):
                self.vector_store = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
            else:
                self.vector_store = None
                
        except Exception as e:
            print(f"Error initializing DocumentStore: {str(e)}")
            raise

    def process_and_store_document(self, 
                                 file_path: str, 
                                 metadata: Optional[Dict] = None,
                                 preprocessing_options: Optional[Dict] = None) -> str:
        """
        Process a document file (PDF/DOCX), extract text, preprocess it, and store in vector DB.
        
        Args:
            file_path (str): Path to the document file
            metadata (Dict, optional): Additional metadata for the document
            preprocessing_options (Dict, optional): Options for text preprocessing
            
        Returns:
            str: Document ID of the stored document
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If there's an error processing the document
        """
        try:
            # Initialize metadata if None
            if metadata is None:
                metadata = {}
                
            # Initialize preprocessing options if None
            if preprocessing_options is None:
                preprocessing_options = {
                    'remove_urls': True,
                    'remove_emails': True,
                    'remove_special_chars': True,
                    'normalize_whitespace': True,
                    'normalize_unicode': True,
                    'preserve_line_breaks': True
                }
            
            # Extract text from document
            raw_text = extract_text_from_document(
                file_path,
                preprocess=False
            )
            
            # Preprocess the extracted text
            processed_text = preprocess_text(raw_text, **preprocessing_options)
            
            # Generate document ID
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Update metadata with file information
            # Convert preprocessing_options dict to string to comply with ChromaDB requirements
            file_metadata = {
                "doc_id": doc_id,
                "timestamp": datetime.now().isoformat(),
                "original_file_path": str(file_path),
                "file_type": os.path.splitext(file_path)[1].lower(),
                "text_length": len(processed_text),
                "preprocessing_options": str(preprocessing_options)  # Convert dict to string
            }
            metadata.update(file_metadata)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(processed_text)
            
            # Create documents for vector store
            documents = [
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_chunk_{i}",
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "file_type": file_metadata["file_type"],
                        "timestamp": file_metadata["timestamp"]
                        # Only include simple metadata that ChromaDB supports
                    }
                )
                for i, chunk in enumerate(chunks)
            ]
            
            # Store in vector database
            if not self.vector_store:
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                self.vector_store.add_documents(documents)
            
            # Persist to disk
            self.vector_store.persist()
            
            return doc_id
            
        except Exception as e:
            raise ValueError(f"Error processing document: {str(e)}")

    def search_text(self, 
                   query: str, 
                   num_results: int = 5,
                   min_score: float = 0.7) -> List[Dict]:
        """
        Enhanced search with context and better scoring
        """
        if not self.vector_store:
            return []
        
        # Get more results initially to filter by score
        initial_results = self.vector_store.similarity_search_with_score(
            query=query,
            k=num_results * 2  # Get more results initially
        )
        
        # Process and combine related chunks
        processed_results = []
        seen_chunks = set()
        
        for doc, score in initial_results:
            doc_id = doc.metadata["doc_id"]
            chunk_index = doc.metadata["chunk_index"]
            
            # Skip if we've already processed this chunk
            if (doc_id, chunk_index) in seen_chunks:
                continue
                
            # Get surrounding chunks for context
            surrounding_chunks = self._get_surrounding_chunks(doc_id, chunk_index)
            
            # Combine content from surrounding chunks
            combined_content = "\n".join(chunk["content"] for chunk in surrounding_chunks)
            
            # Add chunk IDs to seen set
            for chunk in surrounding_chunks:
                seen_chunks.add((doc_id, chunk["metadata"]["chunk_index"]))
            
            processed_results.append({
                "content": combined_content,
                "metadata": doc.metadata,
                "score": float(score),
                "context_chunks": len(surrounding_chunks)
            })
        
        return processed_results[:num_results]

    def _get_surrounding_chunks(self, doc_id: str, chunk_index: int, window: int = 1) -> List[Dict]:
        """Get surrounding chunks for better context"""
        all_chunks = self.get_document(doc_id)
        if not all_chunks:
            return []
        
        # Sort chunks by index
        sorted_chunks = sorted(all_chunks, key=lambda x: x["metadata"]["chunk_index"])
        
        # Find the chunk and its neighbors
        start_idx = max(0, chunk_index - window)
        end_idx = min(len(sorted_chunks), chunk_index + window + 1)
        
        return sorted_chunks[start_idx:end_idx]

    def get_document(self, doc_id: str) -> List[Dict]:
        """
        Retrieve all chunks of a specific document
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            List of document chunks with metadata
        """
        if not self.vector_store:
            return []
        
        # Search for all chunks of the document
        results = self.vector_store.similarity_search_with_score(
            query="",
            k=100,  # Large number to get all chunks
            filter={"doc_id": doc_id}
        )
        
        # Sort chunks by index
        chunks = sorted(
            [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                }
                for doc, score in results
            ],
            key=lambda x: x["metadata"]["chunk_index"]
        )
        
        return chunks 

if __name__ == "__main__":
    doc_store = DocumentStore()
    
    # More specific query
    search_query = "What is the company's SAM.gov registration status and details?"
    results = doc_store.search_text(search_query, num_results=3)
    
    print("\nSearch Results for SAM.gov Registration:")
    print("-" * 70)
    
    found_relevant_info = False
    for i, result in enumerate(results, 1):
        content = result['content'].strip()
        score = result['score']
        
        # Keywords to look for
        keywords = ['sam', 'sam.gov', 'registration', 'cage', 'duns']
        
        # Check if content contains relevant keywords
        if any(keyword in content.lower() for keyword in keywords):
            found_relevant_info = True
            print(f"\nResult {i}:")
            print("Content:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            print(f"Relevance Score: {score:.4f}")
            print(f"Number of context chunks: {result['context_chunks']}")
            print("-" * 70)
    
    if not found_relevant_info:
        print("\nNo clear information about SAM.gov registration found.")
        print("\nSuggestions:")
        print("1. Try searching for 'CAGE Code' or 'DUNS Number' specifically")
        print("2. Search for 'federal registration' or 'government registration'")
        print("3. Check if the document contains this information")