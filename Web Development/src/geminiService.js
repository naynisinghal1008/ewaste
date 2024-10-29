import axios from 'axios';

const API_KEY = 'AIzaSyCu7oifBTo7zDog6VL2NUp7mkom82boBF4';

const geminiService = async (prompt) => {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${API_KEY}`;
  
  const data = {
    contents: [
      {
        parts: [
          {
            text: prompt,
          },
        ],
      },
    ],
  };

  try {
    const response = await axios.post(url, data, {
      headers: {
        'Content-Type': 'application/json',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('Error fetching response from Google API:', error);
    throw error;
  }
};

export default geminiService;