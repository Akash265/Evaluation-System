#load data 
import os
import random 
import itertools
import warnings
from typing import List

from langchain.chains import QAGenerationChain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain
from langchain.prompts import PromptTemplate

os.getenv()


def load_data(filepath:str):
    with open(filepath,'r') as f:
        text_data=' '.join(f.read().splitlines())
    return text_data



def generate_eval(text: str, num_questions: int, chunk: int):
    """
    Generate eval set
    @param text: text to generate eval set from
    @param num_questions: number of questions to generate
    @param chunk: chunk size to draw question from in the doc
    @return: eval set as JSON list
    """
    
    n = len(text)
    starting_indices = [random.randint(0, n - chunk) for _ in range(num_questions)]
    sub_sequences = [text[i:i + chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
        except:
            pass
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full




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

And explain why the my answer is correct or incorrect.
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

    # Evaluate the predictions and ground truth using the evaluation chain
    graded_outputs = eval_chain.evaluate(
        predicted_dataset,
        predictions,
        question_key="question",
        prediction_key="result"
    )

    return graded_outputs
