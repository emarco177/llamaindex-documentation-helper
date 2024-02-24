from dotenv import load_dotenv
import os
from llama_index.core import SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

if __name__ == "__main__":
    print("Going to ingest pinecone documentation...")
    from llama_index.readers.file import UnstructuredReader

    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex-docs",
        file_extractor={".html": UnstructuredReader()},
    )
    documents = dir_reader.load_data()
    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", embed_batch_size=100)

    Settings.llm = OpenAI()
    Settings.embed_model = OpenAIEmbedding()

    index_name = "llamaindex-documentation-helper"
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index(name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        show_progress=True,
    )
    print("finished ingesting...")
