import React, { useState } from 'react';
import geminiService from '../../geminiService';
import { FaPaperPlane } from 'react-icons/fa';

const Chatbot = () => {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleSend = async () => {
    setLoading(true);
    try {
      const apiResponse = await geminiService(input);
      setResponse(apiResponse.candidates[0].content.parts[0].text);
      setInput(''); // Clear input after sending
    } catch (error) {
      setResponse('Error: Could not retrieve response.');
    }
    setLoading(false);
  };

  return (
    <div className="chatbot-container bg-gray-200 p-6 rounded-md shadow-md max-w-2xl mx-auto">
      <h1 className="text-xl font-bold text-orange-800 mb-4 text-center">AI Chatbot</h1>
      <div className="flex items-center mb-4">
        <textarea
          className="flex-grow p-2 border rounded-md resize-none"
          value={input}
          onChange={handleInputChange}
          placeholder="Ask me anything..."
          rows="3"
        />
        <button
          onClick={handleSend}
          className="bg-orange-600 text-white px-4 py-2 ml-2 rounded-md flex items-center"
          disabled={loading}
        >
          <FaPaperPlane className="mr-2" />
          {loading ? 'Sending...' : 'Send'}
        </button>
      </div>

      {response && (
        <div className="response p-4 bg-white rounded-md shadow-sm mt-4">
          <strong>Response:</strong>
          <p className="mt-2">{response}</p>
        </div>
      )}
    </div>
  );
};

export default Chatbot;