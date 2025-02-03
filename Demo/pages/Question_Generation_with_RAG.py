import streamlit as st
from utils import *

st.set_page_config(layout="wide")

inference_api_key = "##"
col1, col2 = st.columns(2)

with open("./style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.title("1. Retrieve the File")
with st.form("Retrieval Form"):
    topic = st.text_input(
        "Enter Your Topic:",
        placeholder="Please, enter a topic to retrieve a context about",
    )
    submitted = st.form_submit_button("Retrieve")

    if submitted:
        if topic:
            # Get the file from the wikipedia
            article_text, success_code = get_wikipedia_article(topic)

            if success_code == 1:
                st.success(f"{topic} File Reterieved from Wikipedia Successfully!")

                with open(f"{topic}.txt", "w", encoding="utf-8") as file:
                    file.write(article_text)

                # Get relevant documents from the file
                with st.spinner(f"Getting Relevant Chunks..."):
                    documents = load_documents(f"{topic}.txt")
                    chunks = split_to_chunks(documents)
                    retriever = retrieve(inference_api_key, chunks)
                    releveant_documents = retriever.get_relevant_documents(topic)

                st.write(releveant_documents)

        else:
            st.error("Please, enter a topic to retreive a context about")


st.title("2. Generate Questions")
with st.form("Generation Form"):
    context = st.text_area(
        label="Enter Your Context: ",
        placeholder="Please, enter a context to generate question from",
        height=300,
        disabled=True,
    )
    answer = st.text_input(
        label="Enter Your Answer",
        placeholder="Please, enter an answer snippet from the provided context",
    )
    num_of_questions = st.number_input(
        label="Enter a Number of Generated Questions:",
        placeholder="Please, enter a number of generated questions you need",
        min_value=1,
        max_value=5,
    )

    submitted = st.form_submit_button("Generate")

    if submitted:
        if context:
            if answer:
                pass

            else:
                st.error("Please, provide an answer snippet")

        else:
            st.error("Please, provide a context to generate questions from")
