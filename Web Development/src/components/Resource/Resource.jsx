import React from 'react';
import { FaYoutube } from 'react-icons/fa';

const Resource = () => {
  const videos = [
    {
      title: 'How E-waste is harming the world?',
      description: 'Get to know about the harmful effects of E-waste',
      link: 'https://youtu.be/-uyIzKIw0xY?si=J3T3xKSfh-Cz81G1',
    },
    {
      title: 'What is E-waste',
      description: 'In order to know about what is e-waste watch the entire video In order to know about what is e-waste watch the entire video In order to know about what is e-waste watch the entire video',
      link: 'https://youtu.be/HQZjouMTH08?si=H_4DOc11KmV-VTc9',
    },
    {
      title: 'The E-Waste Tragedy',
      description: 'A deep dive into the global e-waste crisis and the illicit e-waste tradeA deep dive into the global e-waste crisis and the illicit e-waste trade.',
      link: 'https://www.youtube.com/watch?v=dd_ZttK3PuM',
    },
    {
      title: 'An innovation to eliminate E-waste by NaMo E-waste',
      description: 'To tackle the seriousness of this problem, Akshay Jain started NaMo E-Waste that specialises in e-waste management.',
      link: 'https://youtu.be/q1f_h_Detqg?si=qor6gkJhJCETuBVH',
    },
    {
      title: 'The Story of Electronics',
      description: 'An animated video about the lifecycle of electronics and their environmental impact.An animated video about the lifecycle of electronics and their environmental impact',
      link: 'https://www.youtube.com/watch?v=sW_7i6T_H78',
    },
    {
      title: 'E-waste and its consequences',
      description: 'A short video by National Geographic explaining what e-waste is and its consequences.',
      link: 'https://www.youtube.com/watch?v=9LW6bGHL1uc',
    },
  ];

  return (
    <div className="container mx-auto my-10 p-8 bg-gray-100 rounded-lg shadow-lg">
      <h1 className="text-4xl font-bold text-orange-800 text-center mb-6">Learn About E-Waste</h1>
      <p className="text-lg text-gray-600 text-center mb-12">
        Explore these educational videos to understand the importance of e-waste management and how you can contribute to sustainability.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {videos.map((video, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-md hover:shadow-xl transition-shadow duration-300">
            <h2 className="text-2xl font-bold text-orange-800 mb-4">{video.title}</h2>
            <p className="text-gray-600 mb-4">{video.description}</p>
            <a
              href={video.link}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center text-white bg-orange-800 hover:bg-orange-700 py-2 px-4 rounded-lg transition-colors duration-300"
            >
              <FaYoutube className="mr-2" /> Watch Video
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Resource;