import wikipedia
from transformers import pipeline

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Weaviate

import weaviate
from weaviate.embedded import EmbeddedOptions


def get_wikipedia_article(query):
    try:
        page = wikipedia.page(query)
        return page.content, 1
    
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Ambiguous query. Did you mean one of these: {', '.join(e.options)}", -1
    
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{query}'.", -2


def load_documents(file_name):
    loader = TextLoader(file_name, encoding='utf8')
    documents = loader.load()

    return documents


def split_to_chunks(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    return chunks


def retrieve(inference_api_key, chunks):
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


pipe = pipeline('text2text-generation', model='mohammedaly2222002/t5-small-squad-qg-v2')
