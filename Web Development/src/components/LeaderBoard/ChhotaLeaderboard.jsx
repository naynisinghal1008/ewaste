import React, { useState } from "react";
import { FaArrowLeft, FaArrowRight } from "react-icons/fa";

const ChhotaLeaderboard = () => {
  const [scrollPosition, setScrollPosition] = useState(0);

  const participants = [
    { name: 'Alice Johnson', score: 900, validSubmissions: 45, invalidSubmissions: 5 },
    { name: 'Bob Smith', score: 850, validSubmissions: 40, invalidSubmissions: 10 },
    { name: 'Charlie Brown', score: 800, validSubmissions: 35, invalidSubmissions: 7 },
    { name: 'David Wilson', score: 750, validSubmissions: 30, invalidSubmissions: 8 },
    { name: 'Emma Watson', score: 700, validSubmissions: 25, invalidSubmissions: 6 },
    { name: 'Franklin Pierce', score: 650, validSubmissions: 20, invalidSubmissions: 4 },
    { name: 'Grace Lee', score: 600, validSubmissions: 15, invalidSubmissions: 3 },
    { name: 'Henry Adams', score: 550, validSubmissions: 12, invalidSubmissions: 5 },
    { name: 'Isabella Garcia', score: 500, validSubmissions: 10, invalidSubmissions: 1 },
    { name: 'Jack Brown', score: 450, validSubmissions: 9, invalidSubmissions: 2 },
    { name: 'Katherine Wilson', score: 400, validSubmissions: 8, invalidSubmissions: 1 },
    { name: 'Liam Smith', score: 350, validSubmissions: 7, invalidSubmissions: 0 },
    { name: 'Mia Johnson', score: 300, validSubmissions: 6, invalidSubmissions: 3 },
    { name: 'Noah Davis', score: 250, validSubmissions: 5, invalidSubmissions: 4 },
    { name: 'Olivia Martinez', score: 200, validSubmissions: 4, invalidSubmissions: 5 },
    { name: 'Paul Thompson', score: 150, validSubmissions: 3, invalidSubmissions: 2 },
    { name: 'Quinn Jackson', score: 100, validSubmissions: 2, invalidSubmissions: 1 },
    { name: 'Rachel Green', score: 50, validSubmissions: 1, invalidSubmissions: 0 },
    { name: 'Sam Wilson', score: 25, validSubmissions: 1, invalidSubmissions: 2 },
    { name: 'Tina Turner', score: 10, validSubmissions: 0, invalidSubmissions: 1 },
  ];

  const scroll = (direction) => {
    const container = document.getElementById("leaderboard-container");
    const scrollAmount = 300;
    if (direction === "left") {
      container.scrollBy({ left: -scrollAmount, behavior: "smooth" });
      setScrollPosition(container.scrollLeft - scrollAmount);
    } else {
      container.scrollBy({ left: scrollAmount, behavior: "smooth" });
      setScrollPosition(container.scrollLeft + scrollAmount);
    }
  };

  return (
    <div className="bg-gray-100 p-8">
      <h2 className="text-3xl font-bold text-orange-800 mb-6 text-center">
        To Be Leaders
      </h2>
      <div className="relative">
        <button
          onClick={() => scroll("left")}
          className="absolute left-0 top-1/2 transform -translate-y-1/2 bg-orange-800 text-white rounded-full p-2 shadow-lg"
        >
          <FaArrowLeft />
        </button>
        <div
          id="leaderboard-container"
          className="flex overflow-x-auto scrollbar-hide space-x-4 px-10 py-4 bg-white shadow-lg rounded-lg"
          style={{ scrollSnapType: "x mandatory" }}
        >
          {participants.map((participant, index) => (
            <div
              key={index}
              className="min-w-[250px] max-w-[250px] bg-orange-100 p-4 rounded-lg shadow-md flex-shrink-0"
              style={{ scrollSnapAlign: "start" }}
            >
              <h3 className="text-xl font-semibold text-orange-800 mb-2">
                {participant.name}
              </h3>
              <p className="text-gray-700">
                <strong>Score:</strong> {participant.score}
              </p>
              <p className="text-green-600">
                <strong>Valid Submissions:</strong> {participant.validSubmissions}
              </p>
              <p className="text-red-600">
                <strong>Invalid Submissions:</strong> {participant.invalidSubmissions}
              </p>
            </div>
          ))}
        </div>
        <button
          onClick={() => scroll("right")}
          className="absolute right-0 top-1/2 transform -translate-y-1/2 bg-orange-800 text-white rounded-full p-2 shadow-lg"
        >
          <FaArrowRight />
        </button>
      </div>
    </div>
  );
};

export default ChhotaLeaderboard;