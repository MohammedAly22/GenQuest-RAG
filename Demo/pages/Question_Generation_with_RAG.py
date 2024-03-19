import streamlit as st
import os
from utils import (
    pipe_v1,
    pipe_v2,
    get_wikipedia_article,
    load_documents,
    split_to_chunks,
    retrieve,
    prepare_instruction)

with open('Demo/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
# 1. Retreival Form
st.title('1. Retrieve the Document')
with st.form('Retrieval Form'):
    if 'inference_api_key' not in os.environ:
        inference_api_key = st.text_input('Enter Your Hugging Face Access Token: ', type='password')
        if inference_api_key:
            os.environ['inference_api_key'] = inference_api_key
        
    topic = st.text_input('Enter Your Topic:', placeholder='Please, enter a topic to retrieve a context about')
    submitted = st.form_submit_button('Retrieve')
    
    if submitted:
        if topic:
            # Get the file from the wikipedia
            article_text, success_code = get_wikipedia_article(topic)
            
            if success_code == 1:
                st.success(f'{topic} File Reterieved from Wikipedia Successfully!')
                
                with open(f"{topic}.txt", "w", encoding="utf-8") as file:
                    file.write(article_text)
                
                # Get relevant documents from the file
                with st.spinner(f"Getting Relevant Chunks..."):
                    documents = load_documents(f"{topic}.txt")
                    chunks = split_to_chunks(documents)
                    retriever = retrieve(os.environ['inference_api_key'], chunks)
                    relevant_documents = retriever.get_relevant_documents(topic)

                st.session_state['context'] = relevant_documents[0].page_content
            else:
                st.error(article_text)

        else:
            st.error('Please, enter a topic to retreive a context about')

# 2. Generation Form
st.title('2. Generate Questions')

with st.form('Generation Form'):
    selected_model = st.selectbox(label='Select a Model:', options=['T5-Small-V2', 'T5-Small-V1'])
    context = st.text_area(label='Enter Your Context: ', placeholder='Please, enter a context to generate question from', height=298, key='context')
    answer = st.text_input(label='Enter Your Answer:', placeholder='Please, enter an answer snippet from the retrieved context')
    num_of_questions = st.number_input(
        label='Enter a Number of Generated Questions:',
        placeholder='Please, enter a number of generated questions you need',
        min_value=1,
        max_value=5)

    
    submitted = st.form_submit_button('Generate')

    if submitted:
        if context:
            if answer:
                prompt = prepare_instruction(context, answer)
                with st.spinner(f"Generating Questions..."):
                    if selected_model == 'T5-Small-V1':
                        generated_output = pipe_v1(prompt, num_return_sequences=num_of_questions, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)
                    else:
                        generated_output = pipe_v2(prompt, num_return_sequences=num_of_questions, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)
                        
                st.write('Generated Question(s):')
                for i, item in enumerate(generated_output):
                    st.info(f"Question #{i+1}: {item['generated_text']}")
            else:
                st.error('Please, provide an answer snippet')
        
        else:
            st.error('Please, provide a context to generate questions from')
