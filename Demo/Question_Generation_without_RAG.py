import streamlit as st
from utils import pipe_v1, pipe_v2, prepare_instruction


with open('Demo/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("Question Generation without RAG")
with st.form('Generation Form'):
    selected_model = st.selectbox(label='Select a Model:', options=['T5-Small-V2', 'T5-Small-V1'])
    context = st.text_area(label='Enter Your Context: ', placeholder='Please, enter a context to generate question from', height=200)
    answer = st.text_input(label='Enter Your Answer:', placeholder='Please, enter an answer snippet from the provided context')
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
