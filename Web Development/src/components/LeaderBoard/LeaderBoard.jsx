import React from 'react';
import ChhotaLeaderboard from './ChhotaLeaderboard'; 

const leaderboardData = [
  { name: 'John Doe', score: 150, validSubmissions: 20, invalidSubmissions: 5 },
  { name: 'Jane Smith', score: 140, validSubmissions: 18, invalidSubmissions: 3 },
  { name: 'Michael Lee', score: 130, validSubmissions: 15, invalidSubmissions: 4 },
  { name: 'Alice Johnson', score: 125, validSubmissions: 14, invalidSubmissions: 2 },
  { name: 'Chris Evans', score: 120, validSubmissions: 13, invalidSubmissions: 1 },
  { name: 'Emma Watson', score: 115, validSubmissions: 12, invalidSubmissions: 6 },
  { name: 'Sophia Brown', score: 110, validSubmissions: 10, invalidSubmissions: 2 },
  { name: 'Liam Davis', score: 105, validSubmissions: 9, invalidSubmissions: 3 },
  { name: 'Olivia Wilson', score: 100, validSubmissions: 8, invalidSubmissions: 4 },
  { name: 'Noah Moore', score: 95, validSubmissions: 7, invalidSubmissions: 5 },
];

const LeaderBoard = () => {
  return (
    <div className="container mx-auto my-10 p-5 bg-white shadow-md rounded-lg">
      <h1 className="text-3xl font-bold text-orange-800 mb-6 text-center">Leaderboard</h1>
      <table className="min-w-full bg-white border border-gray-300">
        <thead>
          <tr>
            <th className="py-2 px-4 border-b text-left">Rank</th>
            <th className="py-2 px-4 border-b text-left">Name</th>
            <th className="py-2 px-4 border-b text-left">Score</th>
            <th className="py-2 px-4 border-b text-left">Valid Submissions</th>
            <th className="py-2 px-4 border-b text-left">Invalid Submissions</th>
          </tr>
        </thead>
        <tbody>
          {leaderboardData.map((user, index) => (
            <tr key={index} className="hover:bg-gray-100">
              <td className="py-2 px-4 border-b">{index + 1}</td>
              <td className="py-2 px-4 border-b">{user.name}</td>
              <td className="py-2 px-4 border-b">{user.score}</td>
              <td className="py-2 px-4 border-b">{user.validSubmissions}</td>
              <td className="py-2 px-4 border-b">{user.invalidSubmissions}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-10">
        <ChhotaLeaderboard /> 
      </div>
    </div>
  );
};

export default LeaderBoard;