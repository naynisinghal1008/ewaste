

import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FaMapMarkerAlt } from "react-icons/fa";
import Chatbot from "../Chatbot/Chatbot";

const locationsData = {
  "Delhi": [
    {
      name: "Green E-Waste Recyclers",
      address: "Vasant Kunj",
      id: 1,
      image: "https://imgs.search.brave.com/1MAx9BcasH6FIBZ5alckBaT0rGYIcSLHNaVtSllSqGw/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/Y2hlYXBvZ3VpZGVz/LmNvbS93cC1jb250/ZW50L3VwbG9hZHMv/c2l0ZXMvMi8yMDE3/LzAxL3JlY3ljbGUt/c2hvcC1pU3RvY2st/MTE5NTg4NDI5My03/NzB4NTEzLmpwZw", // Add image URL
    },
    {
      name: "RecycleIT Center",
      address: "Palika Bazar",
      id: 2,
      image: "https://imgs.search.brave.com/iUIXEG2DJdIGG5cxYjNDtL1YwasQzBseEnRQrF2dEFY/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvNDgz/MzgwOTg4L3Bob3Rv/L2dhcmJhZ2UtcmVj/eWNsaW5nLXBsYW50/LmpwZz9zPTYxMng2/MTImdz0wJms9MjAm/Yz1NY1labWZOcF9O/WHpydlk3aUtIWjB4/WDZiZFZ2WGV2M2lk/UWVzUVFSbEZNPQ", // Add image URL
    },
    {
      name: "ropan Center",
      address: "uttam nagar east",
      id: 2,
      image: "https://imgs.search.brave.com/iUIXEG2DJdIGG5cxYjNDtL1YwasQzBseEnRQrF2dEFY/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvNDgz/MzgwOTg4L3Bob3Rv/L2dhcmJhZ2UtcmVj/eWNsaW5nLXBsYW50/LmpwZz9zPTYxMng2/MTImdz0wJms9MjAm/Yz1NY1labWZOcF9O/WHpydlk3aUtIWjB4/WDZiZFZ2WGV2M2lk/UWVzUVFSbEZNPQ", // Add image URL
    },
    {
      name: "ewaste Center",
      address: "nazafgarh",
      id: 2,
      image: "https://imgs.search.brave.com/iUIXEG2DJdIGG5cxYjNDtL1YwasQzBseEnRQrF2dEFY/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvNDgz/MzgwOTg4L3Bob3Rv/L2dhcmJhZ2UtcmVj/eWNsaW5nLXBsYW50/LmpwZz9zPTYxMng2/MTImdz0wJms9MjAm/Yz1NY1labWZOcF9O/WHpydlk3aUtIWjB4/WDZiZFZ2WGV2M2lk/UWVzUVFSbEZNPQ", // Add image URL
    },
  ],
  "Mumbai": [
    {
      name: "E-Waste Management Inc.",
      address: "Dadar",
      id: 3,
      image: "https://example.com/ewaste-management.jpg", 
    },
    {
      name: "Green Earth Solutions",
      address: "Andheri",
      id: 4,
      image: "https://example.com/green-earth.jpg", 
    },
  ],
};

const RecyclingCenters = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedLocation, setSelectedLocation] = useState("");
  const [recyclingCenters, setRecyclingCenters] = useState([]);
  const navigate = useNavigate(); 

  const handleLocationSelect = () => {
    if (locationsData[searchTerm]) {
      setSelectedLocation(searchTerm);
      setRecyclingCenters(locationsData[searchTerm]);
      setSearchTerm(""); 
    } else {
      alert("Please select a valid location from the list.");
    }
  };

  const filteredLocations = Object.keys(locationsData).filter((location) =>
    location.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleCenterClick = (centerId) => {
    navigate("/ImageUpload", { state: { centerId } });
  };

  return (
   <>
    <div className="max-w-2xl mx-auto p-8">
      <h1 className="text-3xl font-bold mb-6 text-center">E-Waste Recycling Centers</h1>

      <input
        type="text"
        placeholder="Enter your location"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        className="w-full px-4 py-2 mb-4 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-orange-500"
      />

      <button
        onClick={handleLocationSelect}
        className="w-full bg-orange-500 text-white py-2 rounded-md hover:bg-orange-600"
      >
        Find Recycling Centers
      </button>

      {searchTerm && (
        <ul className="bg-white border border-gray-300 rounded-md max-h-48 overflow-y-auto">
          {filteredLocations.map((location) => (
            <li
              key={location}
              onClick={() => setSearchTerm(location)} 
              className="px-4 py-2 cursor-pointer hover:bg-gray-100"
            >
              {location}
            </li>
          ))}
        </ul>
      )}

      {selectedLocation && recyclingCenters.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mt-8 mb-4 text-center">
            Recycling Centers in {selectedLocation}
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {recyclingCenters.map((center) => (
              <div
                key={center.id}
                onClick={() => handleCenterClick(center.id)} 
                className="p-6 border border-gray-300 rounded-lg shadow-lg bg-white hover:shadow-2xl hover:bg-blue-50 cursor-pointer transition-all duration-300 ease-in-out transform hover:-translate-y-2"
              >
                <img
                  src={center.image}
                  alt={center.name}
                  className="w-full h-32 object-cover rounded-md mb-4"
                />
                <h3 className="text-xl font-bold text-blue-600">{center.name}</h3>
                <p className="flex items-center mt-2 text-gray-700">
                  <FaMapMarkerAlt className="mr-2 text-red-500" />
                  {center.address}
                </p>
                <p className="mt-4 text-sm text-gray-500">Click for more details</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {selectedLocation && recyclingCenters.length === 0 && (
        <p className="text-center mt-4 text-red-500">
          No recycling centers found for {selectedLocation}.
        </p>
      )}
    </div>
    <Chatbot></Chatbot>
   </>
  );
};

export default RecyclingCenters;









