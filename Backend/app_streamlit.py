import streamlit as st
from langchain_file import generate_eval, grade_model_answer
import json
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key using os.environ
api_key = os.environ.get("OPEN_API_KEY")

# Initialize session state
if 'eval_set' not in st.session_state:
    st.session_state.eval_set = None
    st.session_state.num_of_questions = 0
    st.session_state.textbox_content = []

def extract_text():
    uploaded_file = st.file_uploader("Upload a Markdown file", type=["md"])
    
    question_types = ["Multiple Choice Question (MCQ)", "Coding", "Theoretical"]
    selected_question_type = st.selectbox("Select the type of question:", question_types)

    num_of_questions = st.number_input("Enter the number of questions:", min_value=1, value=1)
    submit_questions_button = st.button("Submit")

    if submit_questions_button:
        if uploaded_file is not None:
            # Read the content of the Markdown file
            eval_set={"questions":[],"answers":[]}
            text = uploaded_file.read().decode('utf-8')

            st.session_state.eval_set = generate_eval(text, num_questions=num_of_questions, chunk=3000,question_type=selected_question_type)
            st.session_state.num_of_questions = num_of_questions
            st.session_state.selected_question_type = selected_question_type
            with open("data.json", "w") as json_file:
                json.dump(st.session_state.eval_set, json_file)


def main():
    st.title("Concept Evaluation System")

    st.header("Welcome to Concept Evaluation System")
    st.write("Upload a Markdown file and evaluate the answers.")

    extract_text()

    if st.session_state.eval_set is not None:
        st.write("Generated Questions:")
        for i in range(st.session_state.num_of_questions):
            if st.session_state.selected_question_type=="Multiple Choice Question (MCQ)":
                st.write(f"Question {i + 1}: {st.session_state.eval_set[i]['question']}")
                st.write(f"Option A: {st.session_state.eval_set[i]['A']}")
                st.write(f"Option B: {st.session_state.eval_set[i]['B']}")
                st.write(f"Option C: {st.session_state.eval_set[i]['C']}")
                st.write(f"Option D: {st.session_state.eval_set[i]['D']}")
            else:
                st.write(f"Question {i + 1}: {st.session_state.eval_set[i]['question']}")
                
                

        st.write("Enter answers:")
        st.session_state.textbox_content = []
        
        for i in range(st.session_state.num_of_questions):
            answer = st.text_input(f"Answer {i + 1}")
            st.session_state.textbox_content.append(answer)

        submit_answers_button = st.button("Submit Answers")
        
        if submit_answers_button:
            st.write("Submitted Answers:")
            for i, answer in enumerate(st.session_state.textbox_content):
                st.write(f"Answer {i + 1}: {answer}")

            answer_set = [{'result': text} for text in st.session_state.textbox_content]
          
            output = grade_model_answer(st.session_state.eval_set, answer_set)
            st.write("Grades and Explanations:")
            for i, entry in enumerate(output):
                result = entry['results']
                grade = 'Correct' if 'GRADE: CORRECT' in result else 'Incorrect'
                explanation = result.split('\n\n')[1]

                st.write(f"Grade {i+1}: {grade}")
                if grade == 'Incorrect':
                    st.write(explanation)

        

if __name__ == "__main__":
    main()
