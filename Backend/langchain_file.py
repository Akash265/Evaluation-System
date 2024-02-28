#load data 
import os
import random 
import itertools
import warnings
from typing import List

from langchain.chains import QAGenerationChain
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,MarkdownTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema,StructuredOutputParser
from langchain.chains import LLMChain
import json
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

os.getenv('OPENAI_API_KEY')


def load_data(filepath:str):
    with open(filepath,'r') as f:
        text_data=' '.join(f.read().splitlines())
    return text_data
def parse(response):
    try:
        split_response = response.split("QUESTION:")[1]
    except:
        split_response = response.split("MCQ:")[1]
    question = split_response.split("ANSWER:")[0].strip()
    answer = split_response.split("ANSWER:")[1].strip()

    # Creating a dictionary
    response_dict = {
        "question": question,
        "answer": answer
    }
    return response_dict

def split_text(content):
    # First we make our text splitter
    text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=200)

    # Then we split our user information into different documents
    docs = text_splitter.create_documents([content])

    return docs

teacher_prompt_template = """You are the teacher in a quiz setting, and the student has requested question in detail for evaluation based on the text provided. The student can choose the type of question: coding, MCQ (multiple-choice), or theoretical. Provide a suitable question and its answer based on the chosen type.Please focus on the type of question asked , it is important to give exactly that kind of question as specified by the student.


TEXT: {text} # text on which the question will be based on 
QUESTION TYPE: {question_type}  # Specify 'coding', 'mcq', or 'theoretical'

# Based on the chosen question type, generate a question and answer.
If the question type is coding also write the code you are refrencing in the question or with which the question is related to. 
QUESTION: question
ANSWER: answer

"""
qa_prompt_template="""You are a smart assistant designed to help teachers come up with detailed {question_type} question.
Given a topic, you must come up with a detailed {question_type} question that can be used to test a student's understanding of the topic .
Please focus on the type of question asked, that is detailed {question_type} question, only generate a single detailed {question_type} question based on the topic.
When coming up with this {question_type} question and answer, you must respond in the following format:

QUESTION: DETAILED_{question_type}_QUESTION_HERE,
ANSWER: THE_ANSWER_HERE

Please come up with a single detailed {question_type} question and answer ,in a dictionary format ,from the following topic:
----------------
TOPIC TITLE: {topic_title}
TOPIC DESCRIPTION: {topic_description}"""




# Topic output parser
class Topic(BaseModel):
    title: List = Field(description="List of {n} titles of the topics extracted from the provided markdown text")
    description: List = Field(description="List of {n} descripitons of the topics extracted from the provided markdown text")

topic_output_parser = JsonOutputParser(pydantic_object=Topic)
topic_format_instructions = topic_output_parser.get_format_instructions()

# MCQ output parser
class MCQ(BaseModel):
    question: str = Field(description="The MCQ Question")
    A:str =Field(description="The first option of the MCQ")
    B:str =Field(description="The second option of the MCQ")
    C:str =Field(description="The third option of the MCQ")
    D:str =Field(description="The fourth option of the MCQ")
    answer:str = Field(description="Answer of the MCQ Question")

MCQ_output_parser = JsonOutputParser(pydantic_object=MCQ)
MCQ_format_instructions = MCQ_output_parser.get_format_instructions()

# Coding output parser
class Coding(BaseModel):
    question: str = Field(description="The Coding Question along with the code (if required)")
    answer:str = Field(description="Code answer in language specified in the question")
    
Coding_output_parser = JsonOutputParser(pydantic_object=Coding)
Coding_format_instructions = Coding_output_parser.get_format_instructions()

# Theoritical output parser
class Theoritical(BaseModel):
    question: str = Field(description="The Theoritical Question based on the topic")
    answer:str = Field(description="The Answer to the theoritical question")
    
Theoritical_output_parser = JsonOutputParser(pydantic_object=Theoritical)
Theoritical_format_instructions = Theoritical_output_parser.get_format_instructions()


topic_prompt="""
Extract {n} educational topics covered in the provided markdown text from which questions can be formulated.
Generate {n} educational topics based on the following text:

{text_input}

{format_instructions}
"""
qa_prompt="""
You are a smart assistant designed to help teachers come up with detailed {question_type} questions.
Given a topic, your task is to generate a detailed {question_type} question aimed at testing a student's understanding of the topic.
Please focus on crafting a single detailed {question_type} question along with its answer, presented in a dictionary format, based on the provided topic:

----------------
TOPIC TITLE: {topic_title}
TOPIC DESCRIPTION: {topic_description}

{format_instructions}
"""

                    
topic_extraction_template = PromptTemplate( input_variables=["text_input","n"] ,partial_variables={"format_instructions":topic_format_instructions},template=topic_prompt)



def generate_eval(text: str, num_questions: int, chunk: int,question_type: str):
    """
    Generate eval set
    @param text: text to generate eval set from
    @param num_questions: number of questions to generate
    @param chunk: chunk size to draw question from in the doc
    @return: eval set as JSON list
    """
    #length of text
    n = len(text)
    if n < chunk:
        raise ValueError("Text length should be greater than or equal to the chunk size.")
    #split markdown docs
    markdown_docs=split_text(text)
    #initialize LLM chain
    llm = OpenAI(temperature=0, max_tokens=2000)
    chain = topic_extraction_template | llm | topic_output_parser
    #Invoke chain
    topics=chain.invoke({"text_input":markdown_docs,
                "n":num_questions})
    
    print("***********************************************************")
    print(topics)
    print("***********************************************************")
    
    
    
    if question_type=="Multiple Choice Question (MCQ)":
        QA_PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["topics","question_type","topic_title","topic_description"],
        partial_variables={"format_instructions":MCQ_format_instructions},
        template=qa_prompt)
        output_parser=MCQ_output_parser
    elif question_type=="Coding":
        QA_PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["topics","question_type","topic_title","topic_description"],
        partial_variables={"format_instructions":Coding_format_instructions},
        template=qa_prompt)
        output_parser=Coding_output_parser
    else:
        QA_PROMPT_TEMPLATE = PromptTemplate(
        input_variables=["topics","question_type","topic_title","topic_description"],
        partial_variables={"format_instructions":Theoritical_format_instructions},
        template=qa_prompt)
        output_parser=Theoritical_output_parser
    
    llm=OpenAI(name='gpt-4')
    #initialize chain
    chain = QA_PROMPT_TEMPLATE | llm | output_parser
    eval_set=[]
    
    #Initializing QAGeneration chain
    for i in range(num_questions):
        # Generate a cool name based on the template
        raw_output=chain.invoke({"topic_title":topics["title"][i],"topic_description":topics["description"][i],"question_type":question_type})
        #response=chain.run(topic_title=topics["topic_name"][i],topic_description=topics["topic_description"][i],question_type=question_type)
        #parsed_response=parse(response)
        eval_set.append(raw_output)
        
    print(eval_set)
    return eval_set



def split_texts(text, chunk_size: int, overlap):
    """
    Split text into chunks
    @param text: text to split
    @param chunk_size:
    @param overlap:
    @param split_method:
    @return: list of str splits
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)
    split_text = text_splitter.split_text(text)
    return split_text





def make_retriever(splits, num_neighbors, _llm):
    """
    Make document retriever
    @param splits: list of str splits
    @param retriever_type: retriever type
    @param embedding_type: embedding type
    @param num_neighbors: number of neighbors for retrieval
    @param _llm: model
    @return: retriever
    """

    # Set embeddings
    embedding = OpenAIEmbeddings()

      
    # retriever
    try:
        vector_store = FAISS.from_texts(splits, embedding)
    except ValueError:
        warnings("`Error using OpenAI embeddings (disallowed TikToken token in the text). Using HuggingFace.`",
                    icon="⚠️")
        vector_store = FAISS.from_texts(splits, HuggingFaceEmbeddings())
    retriever_obj = vector_store.as_retriever(k=num_neighbors)
    return retriever_obj



def make_chain(llm, retriever) -> RetrievalQA:
    """
    Make chain
    @param llm: model
    @param retriever: retriever
    @param retriever_type: retriever type
    @return: chain (or return retriever for Llama-Index)
    """
    qa = RetrievalQA.from_chain_type(llm,chain_type="stuff",
                                        retriever=retriever,
                                        input_key="question")
    return qa


template = """You are a teacher grading a quiz. 
You are given a question, my answer, and the true answer, and are asked to score  my answer as either CORRECT or INCORRECT.

Example Format:
QUESTION: question here
MY ANSWER: my answer here
TRUE ANSWER: true answer here
GRADE: CORRECT or INCORRECT here

Grade my answers based ONLY on their factual accuracy. Ignore differences in punctuation and phrasing between my answer and true answer. It is OK if my answer contains more information than the true answer, as long as it does not contain any conflicting statements. Begin! 

QUESTION: {query}
MY ANSWER: {result}
TRUE ANSWER: {answer}
GRADE:

And explain in detail why the my answer is correct or incorrect.
"""

GRADE_ANSWER_PROMPT = PromptTemplate(input_variables=["query", "result", "answer"], template=template)

def grade_model_answer(predicted_dataset: List, predictions: List) -> List:
    """
    Grades the distilled answer based on ground truth and model predictions.
    @param predicted_dataset: A list of dictionaries containing ground truth questions and answers.
    @param predictions: A list of dictionaries containing model predictions for the questions.
    @param grade_answer_prompt: The prompt level for the grading. Either "Fast" or "Full".
    @return: A list of scores for the distilled answers.
    """

    # Set the grading prompt based on the grade_answer_prompt parameter
    prompt = GRADE_ANSWER_PROMPT

    # Create an evaluation chain
    eval_chain = QAEvalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        prompt=prompt
    )
    for pred in predictions:
        if not pred['result']:
            pred['result'] = 'No answer provided'

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )

    return graded_outputs
