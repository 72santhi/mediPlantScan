import React, { useState } from 'react';

const Feedback = () => {
  const [feedback, setFeedback] = useState('');
  const [report, setReport] = useState('');
  const [suggestions, setSuggestions] = useState('');

  const handleSubmit = (event) => {
    event.preventDefault();
    // Here you can handle the submission of feedback, report, and suggestions
    // You can send this data to your backend server for processing
    console.log('Feedback:', feedback);
    console.log('Report:', report);
    console.log('Suggestions:', suggestions);
    // Optionally, you can reset the form fields after submission
    setFeedback('');
    setReport('');
    setSuggestions('');
  };

  return (
    <div className="feedback-container">
      <h2>Feedback Form</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="feedback">Feedback:</label>
          <textarea
            id="feedback"
            value={feedback}
            onChange={(event) => setFeedback(event.target.value)}
            placeholder="Enter your feedback.."
            rows="4"
          ></textarea>
        </div>
        <div className="form-group">
          <label htmlFor="report">Report Issue:</label>
          <textarea
            id="report"
            value={report}
            onChange={(event) => setReport(event.target.value)}
            placeholder="If there is an issue please report or else leave it blank.."
            rows="4"
          ></textarea>
        </div>
        <div className="form-group">
          <label htmlFor="suggestions">Suggestions:</label>
          <textarea
            id="suggestions"
            value={suggestions}
            onChange={(event) => setSuggestions(event.target.value)}
            placeholder="Enter your suggestions.."
            rows="4"
          ></textarea>
        </div>
        <button className="submit-button" type="submit">Submit</button>
      </form>
    </div>
  );
};

export default Feedback;
