import wikipedia
from transformers import pipeline

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Weaviate

import weaviate
from weaviate.embedded import EmbeddedOptions


def get_wikipedia_article(query):
    """
    Fetch a Wikipedia article based on a query.

    Parameters:
        - query (str): The query to search for on Wikipedia.

    Returns:
        - tuple: A tuple containing the content of the Wikipedia article and a status code.
        If the query is ambiguous, the status code is -1 and a message with possible options is returned.
        If no Wikipedia page is found, the status code is -2 and a message indicating this is returned.
        If the query is successful, the status code is 1 and the content of the article is returned.
    """
    
    try:
        page = wikipedia.page(query)
        return page.content, 1
    
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Did you mean one of these: {', '.join(e.options)}", -1
    
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'.", -2


def load_documents(file_name):
    """
    Load documents from a file using the `TextLoader` from langchain.

    Parameters:
        - file_name (str): The name of the file to load documents from.

    Returns:
        - list: A list of documents loaded from the file.
    """
    
    loader = TextLoader(file_name, encoding='utf8')
    documents = loader.load()

    return documents


def split_to_chunks(documents):
    """
    Split documents into chunks using the `CharacterTextSplitter` from langchain.

    Parameters:
        - documents (list): A list of documents to split into chunks.

    Returns:
        - list: A list of chunks created from the documents.
    """
    
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    return chunks


def retrieve(inference_api_key, chunks):
    """
    Retrieve embeddings for chunks using `HuggingFaceInferenceAPIEmbeddings` and Weaviate.

    Parameters:
        - inference_api_key (str): The API key for HuggingFace's inference API.
    -     chunks (list): A list of chunks to retrieve embeddings for.

    Returns:
        - Weaviate.as_retriever: A retriever object that can be used to retrieve similar chunks.
    """
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2")

    client = weaviate.Client(embedded_options=EmbeddedOptions())

    vectorstore = Weaviate.from_documents(
        client=client,
        documents=chunks,
        embedding=embeddings,
        by_text=False
    )

    return vectorstore.as_retriever()


def prepare_instruction(context, answer):
    """
    Prepare an instruction prompt for a question-answering task.

    Parameters:
        - context (str): The context from which to generate the question.
        - answer (str): The answer to highlight in the question.

    Returns:
        - str: The instruction prompt for the question-answering task.
    """
    
    context_splits = context.split(answer)
    
    text = ""
    for split in context_splits:
        text += split
        text += ' <h> '
        text += answer
        text += ' <h> '
        text += split
    
    instruction_prompt = f"""Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks.
    context:
    ```
    {text}
    ```
    """
    
    return instruction_prompt


pipe_v1 = pipeline('text2text-generation', model='mohammedaly22/t5-small-squad-qg')
pipe_v2 = pipeline('text2text-generation', model='mohammedaly22/t5-small-squad-qg-v2')
