// main.js
function generateTextboxes(numTextboxes) {
    var container = document.getElementById('textboxesContainer');
    container.innerHTML = ''; // Clear previous content
    
    for (var i = 0; i < numTextboxes; i++) {
      var heading = document.createElement('h4');
      heading.textContent = 'Answer ' + (i + 1);
      container.appendChild(heading);
  
      var textbox = document.createElement('input');
      textbox.type = 'text';
      textbox.name = 'textbox_' + (i + 1);
      textbox.className = 'bigTextbox'; // Add a class for styling
      textbox.style.width = '100%';
      container.appendChild(textbox);
      container.appendChild(document.createElement('br'));
    }
    document.getElementById('submit-button').style.display = 'block';
  }
  
  function getQuestions() {
    var numberOfQuestions = parseInt(document.getElementById('numberOfQuestions').value);
    var files = document.getElementById("text_file").files;
    var formData = new FormData();
    var endpoint = '/api/v1/extract_text';
  
    formData.append('text', files[0]);
    formData.append('numofques', numberOfQuestions);
  
    $.ajax({
      type: 'POST',
      url: endpoint,
      data: formData,
      contentType: false,
      cache: false,
      processData: false,
      success: function(response) {
        displayQuestions(response);
      }
    });
    generateTextboxes(numberOfQuestions);
  }
  
  function sendTextBoxContent() {
    var textboxes = document.querySelectorAll('.bigTextbox');
    var content = [];
  
    textboxes.forEach(function(textbox) {
      var textboxContent = textbox.value;
      content.push(textboxContent);
    });
  
    var jsonData = {
      'textboxContent': content
    };
  
    $.ajax({
      type: 'POST',
      url: '/api/v1/send_textbox_content',
      contentType: 'application/json',
      data: JSON.stringify(jsonData),
      success: function(response) {
        displayGradesAndExplanations(response);
      },
      error: function(error) {
        console.error('Error sending data:', error);
      }
    });
  }
  
  document.getElementById('submit-button').addEventListener('click', function() {
    sendTextBoxContent();
  });
  
  function displayQuestions(response) {
    for (let i = 0; i < response.length; i++) {
      let div = document.createElement('div');
      div.innerHTML = `<p><h5>Question ${i + 1}:</h5> ${response[i].question}<br></p>`;
      document.getElementById('json-display').appendChild(div);
    }
  }
  
  function displayGradesAndExplanations(data) {
    var contentDisplay = document.getElementById('content-display');
    contentDisplay.innerHTML = ''; // Clear previous content
  
    data.forEach(function(entry, index) {
      var result = entry.results;
      var grade = result.includes('GRADE: CORRECT') ? 'Correct' : 'Incorrect';
      var explanation = result.split('\n\n')[1];
  
      var div = document.createElement('div');
      div.classList.add('answer');
      div.innerHTML = `
        <h3>Answer ${index + 1}</h3>
        <p><strong>Grade:</strong> ${grade}</p>
      `;
  
      contentDisplay.appendChild(div);
  
      if (grade === 'Incorrect') {
        var explanationDiv = document.createElement('div');
        explanationDiv.innerHTML = `
          <p><strong>Explanation:</strong> ${explanation}</p>
        `;
        contentDisplay.appendChild(explanationDiv);
      }
    });
  }
  