# GenQuest-RAG | Retrieval Augmented Generation (RAG) for Question Generation
**GenQuest-RAG** is an innovative system designed to generate insightful questions by harnessing the power of machine learning and utilizing Wikipedia as a vast knowledge repository. Through the integration of advanced natural language processing techniques, the system can retrieve relevant information from Wikipedia articles and utilize it to augment question generation.

By leveraging RAG, the project goes beyond traditional question-generation approaches by incorporating **contextually rich information extracted from Wikipedia articles**. This enables the system to produce more **accurate**, **contextually relevant**, and **informative** questions across a wide range of topics.

The project aims to enhance learning, comprehension, and knowledge acquisition by providing users with dynamically generated questions tailored to specific topics or areas of interest. Whether used in educational settings, content creation, or research, the Question Generation with RAG project offers a powerful tool for generating high-quality questions that stimulate critical thinking and deepen understanding.

# Tools Used
GenQuest-RAG is implemented using the following Python packages:

| Package | Description |
| --- | --- |
| Wikipedia | A Wikipedia API to make it my knowledge source for Retrieval Augmented Generation (RAG)  |
| PyTorch | An Open-source machine learning framework |
| Transformers | A Hugging Face package contains state-of-the-art Natural Language Processing models |
| Datasets | A Hugging Face package contains popular open-source datasets |
| Evaluate | A Hugging Face package contains several evaluation metrics like BLEU, ROUGE, METEOR, BERTScore, etc. |
| LangChain | A framework for developing applications powered by language models |
| Weaviate | An open-source, cloud-native, vector search engine that allows for semantic search and exploration of structured and unstructured data. |

# Usage
## Running Demo:


## Usage as a high-level Pipeline:
1. Define some useful functions for highlighting the answer in the paragraph and preparing the instruction prompt that will be fed to the model: 
```Python
def highlight_answer(context, answer):
    context_splits = context.split(answer)
    
    text = ""
    for split in context_splits:
        text += split
        text += ' <h> '
        text += answer
        text += ' <h> '
        text += split
    
    return text

def prepare_instruction(answer_highlighted_context):
    instruction_prompt = f"""Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks.
    context:
    ```
    {answer_highlighted_context}
    ```
    """
    
    return instruction_prompt
```
2. Use the model as a Hugging Face Pipeline:
```Python
from transformers import pipeline

pipe = pipeline('text2text-generation', model='mohammedaly2222002/t5-small-squad-qg')

context = """During the 2011–12 season, he set the La Liga and European records\
for most goals scored in a single season, while establishing himself as Barcelona's\
all-time top scorer. The following two seasons, Messi finished second for the Ballon\
d'Or behind Cristiano Ronaldo (his perceived career rival), before regaining his best\
form during the 2014–15 campaign, becoming the all-time top scorer in La Liga and \
leading Barcelona to a historic second treble, after which he was awarded a fifth \
Ballon d'Or in 2015. Messi assumed captaincy of Barcelona in 2018, and won a record \
sixth Ballon d'Or in 2019. Out of contract, he signed for French club Paris Saint-Germain\
in August 2021, spending two seasons at the club and winning Ligue 1 twice. Messi \
joined American club Inter Miami in July 2023, winning the Leagues Cup in August of that year.
"""

answer_highlighted_context = highlight_answer(context=context, answer='Inter Miami')
prompt = prepare_instruction(answer_highlighted_context)
```
This will be the final prompt:
```
Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks
context:
```During the 2011–12 season, he set the La Liga and European records\
for most goals scored in a single season, while establishing himself as Barcelona's\
all-time top scorer. The following two seasons, Messi finished second for the Ballon\
d'Or behind Cristiano Ronaldo (his perceived career rival), before regaining his best\
form during the 2014–15 campaign, becoming the all-time top scorer in La Liga and \
leading Barcelona to a historic second treble, after which he was awarded a fifth \
Ballon d'Or in 2015. Messi assumed captaincy of Barcelona in 2018, and won a record\
 sixth Ballon d'Or in 2019. Out of contract, he signed for French club Paris Saint-Germain\
in August 2021, spending two seasons at the club and winning Ligue 1 twice. Messi \
joined American club  <h> Inter Miami <h> in July 2023, winning the Leagues Cup in August of that year.```
```
3. Use the loaded `pipeline` to generate questions their answer is `Inter Miami`:
```Python
outputs = pipe(prompt, num_return_sequences=3, num_beams=5, num_beam_groups=5, diversity_penalty=1.0)
for output in outputs:
    print(output['generated_text'])
```
Result:
```
1. What club did Messi join in the 2023 season?
2. What was Messi's name of the club that won the Leagues Cup on July 20?
3. What club did Messi join in the Leagues Cup in July 2023?
```

# Dataset
**The Stanford Question Answering Dataset (SQuAD)** is a popular benchmark dataset in the field of natural language processing (NLP) and machine reading comprehension. It was developed by researchers at Stanford University. SQuAD consists of a large collection of real questions posed by crowd workers on a set of Wikipedia articles, where each question is paired with a corresponding passage from the article, and the answer to each question is a segment of text from the corresponding passage.

The goal of SQuAD is to train and evaluate machine learning models to understand and answer questions posed in natural language. It has been widely used as a benchmark for evaluating the performance of various question-answering systems and models, including both rule-based systems and deep learning-based approaches such as neural network models.

# Methodology
## Dataset Preparation
1. I followed Chan and Fan (2019) by introducing the highlight token `<h>` to take into account an answer `a` within context `c` as below:
$x = [ c_1, ..., \lt h\gt , a_1, ..., a_a, \lt h\gt , ..., c_c ]$

2. Preparing the instruction prompt by following this template
```
Generate a question whose answer is highlighted by <h> from the context delimited by the triple backticks.
context:
```{answer_highlighted_context}```
```

## Retrieval Augmented Generation (RAG)
![RAG-Diagram (1)](https://github.com/MohammedAly22/GenQuest-RAG/assets/90681796/46533610-38ae-45e2-9eba-f650ab0a2646)


# Results
I conducted full fine-tuning on two instances of the `t5-small` model, each with differing hyperparameters. Provided below are the detailed `TrainingArguments` for both versions:

| Model           | HyperParameter             | Value |
| ---             | ---                        | ---   |
| T5-Small-FFT-V1 | epochs                     | 3     |
|                 | batch size                 | 32    |
|                 | warmup steps               | 500   |
|                 | weight decay               | 0.01  |
| T5-Small-FFT-V2 | epochs                     | 10    |
|                 | batch size                 | 16    |
|                 | gradient accumlation steps | 4     |
|                 | learning rate              | 5e-5  |
|                 | save total limit           | 2     |
|                 | warmup steps               | 1000  |
|                 | fp16                       | True  |
|                 | weight decay               | 0.01  |

Here are the evaluation metrics for the two versions:
![results](https://github.com/MohammedAly22/GenQuest-RAG/assets/90681796/190cc728-8478-4ebc-ae35-f41f74773104)

