from flask import Flask, request, jsonify, send_from_directory
from langchain_file import generate_eval, grade_model_answer
import json

app = Flask(__name__, static_folder='../Frontend', static_url_path='')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/v1/extract_text', methods=['POST'])
def extract_text():
    received_data = request.files['text']
    numofques = int(request.form['numofques'])

    if received_data.filename != '':
        text = received_data.read().decode('utf-8')
 
    eval_set = generate_eval(text, num_questions=numofques, chunk=3000)

    with open("data.json", "w") as jsonFile:
        json.dump(eval_set, jsonFile)

    return jsonify(eval_set)

@app.route('/api/v1/send_textbox_content', methods=['POST'])
def receive_textbox_content():
    data = request.json
    textbox_content = data.get('textboxContent', [])

    with open('data.json') as f:
        eval_set = json.load(f)

    answer_set = [{'result': text} for text in textbox_content]
    output = grade_model_answer(eval_set, answer_set)

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
