import streamlit as st
from utils import pipe, get_wikipedia_article, load_documents, split_to_chunks, retrieve

# with open("/style.css") as f:
#     st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.title("Question Generation without RAG")
# inference_api_key = "#"
inference_api_key = st.text_input("Enter HF Pass")

# Initialize session state
if "option" not in st.session_state:
    st.session_state["option"] = "Enter a context"


def update_session_state():
    st.session_state["option"] = st.session_state["hidden_option"]


option = st.selectbox(
    "select an option",
    ("Enter a context", "Enter a topic"),
    key="hidden_option",
    on_change=update_session_state,
)
selected_option = None



with st.form("submission_form"):
    if st.session_state["option"] == "Enter a context":
        context = st.text_area("Enter your context", height=300)
        selected_option = "context"

        number_of_questions = st.number_input(
            f"Choose number of questions", min_value=1, max_value=5
        )

    else:
        topic = st.text_input("Enter your topic")
        selected_option = "topic"

        number_of_questions = st.number_input(
            f"Choose number of questions", min_value=1, max_value=5
        )

    submitted = st.form_submit_button("Submit")

    if submitted:
        if selected_option == "context":
            if context != "":
                with st.spinner(f"Generating Questions..."):
                    generated_output = pipe(
                        context,
                        num_return_sequences=number_of_questions,
                        num_beams=5,
                        num_beam_groups=5,
                        diversity_penalty=1.0,
                    )
                st.write("Generated Question")
                for i, item in enumerate(generated_output):
                    st.info(f"Question #{i+1}: {item['generated_text']}")
            else:
                st.error("Please, Enter a Context to Generate From")
        else:
            if topic != "":
                article_text, success_code = get_wikipedia_article(topic)

                if success_code == 1:
                    st.success(f"{topic} File Reterieved Successfully!")

                    with open(f"{topic}.txt", "w", encoding="utf-8") as file:
                        file.write(article_text)

                    with st.spinner(f"Generating Questions..."):
                        documents = load_documents(f"{topic}.txt")
                        chunks = split_to_chunks(documents)
                        retriever = retrieve(inference_api_key, chunks)
                        context = retriever.get_relevant_documents(topic)[
                            0
                        ].page_content
                        # st.write(f'Retrieved Context')
                        # st.info(context)
                        generated_output = pipe(
                            context,
                            num_return_sequences=number_of_questions,
                            num_beams=5,
                            num_beam_groups=5,
                            diversity_penalty=1.0,
                        )

                    st.write("Generated Question")
                    for i, item in enumerate(generated_output):
                        st.info(f"Question #{i+1}: {item['generated_text']}")
                else:
                    st.error(article_text)
            else:
                st.error("Please, Enter a Topic to Retrieve a File About")
