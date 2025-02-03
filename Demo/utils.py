import wikipedia
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from transformers import pipeline


def get_wikipedia_article(query):
    """Fetch Wikipedia article content based on the query."""
    try:
        page = wikipedia.page(query)
        return page.content, 1
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Did you mean one of these: {', '.join(e.options)}", -1
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'.", -2


def load_documents(file_name):
    """Load text documents from a file."""
    loader = TextLoader(file_name, encoding="utf8")
    return loader.load()


def split_to_chunks(documents):
    """Split text documents into smaller chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)


def retrieve(inference_api_key, chunks):
    """Retrieve similar text chunks using Weaviate and embeddings."""
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    # Initialize Weaviate client
    client = weaviate.WeaviateClient(embedded_options=EmbeddedOptions())

    # Create a Weaviate vector store
    vectorstore = Weaviate(client=client, embedding=embeddings)

    # Add documents to Weaviate
    vectorstore.add_documents(chunks)

    return vectorstore.as_retriever()


pipe = pipeline('text2text-generation', model='mohammedaly2222002/t5-small-squad-qg')
