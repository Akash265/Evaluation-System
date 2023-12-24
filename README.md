# Concept Evaluation System

This repository contains the frontend and backend code for a Concept Evaluation System. The system allows users to upload a text file, answer questions related to the text content, and receive evaluations on their answers.

## Frontend

The frontend part of the application is built using HTML, JavaScript, and CSS. It includes the user interface for uploading files, displaying questions, answering questions through textboxes, and receiving evaluations.

### Files
- `index.html`: Main HTML file containing the UI structure.
- `styles.css`: CSS file for styling the UI components.
- `script.js`: JavaScript file handling user interactions, AJAX requests, and DOM manipulation.

### Usage
1. Clone or download this repository.
2. Open `index.html` in a web browser to use the Concept Evaluation System.

## Backend

The backend part of the application is built using Flask, a Python web framework. It handles file processing, question extraction, grading of answers, and communicates with the frontend through API endpoints.

### Files
- `app.py`: Main Flask application file containing backend logic.
- `langchain_file.py`: Supporting Python file for language processing and evaluation.

### Setup
1. Ensure Python is installed on your machine.
2. Install requirements using `pip install -r requirements.txt`
3. Run the Flask application by executing `python app.py` in the terminal.

### Endpoints
- `/`: Serves the main HTML file and frontend interface.
- `/api/v1/extract_text`: Endpoint for extracting text from uploaded files and generating questions.
- `/api/v1/send_textbox_content`: Endpoint for receiving and grading user-provided answers.

## Contributing

Feel free to contribute by submitting bug fixes, feature enhancements, or additional functionality through pull requests.



