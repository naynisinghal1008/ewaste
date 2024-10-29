import React from 'react';

const EWasteTrackingPage = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white shadow-lg rounded-lg p-6 w-full max-w-4xl">
        {/* Header */}
        <div className="flex justify-between items-center pb-4 border-b border-gray-200">
          <div>
            <h1 className="text-2xl font-bold text-gray-800">E-Waste Submission ID: EW12345678</h1>
            <p className="text-gray-500">Track your e-waste submission status and recycling process.</p>
          </div>
        </div>

        {/* Tracking Status */}
        <div className="flex justify-between items-center py-4">
          <div className="flex space-x-4">
            <div className="flex flex-col items-center">
              <div className="bg-orange-500 text-white p-2 rounded-full">
                <i className="fas fa-check"></i> {/* Icon */}
              </div>
              <p className="text-gray-700 font-semibold mt-2">Request Made</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-orange-500 text-white p-2 rounded-full">
                <i className="fas fa-check"></i>
              </div>
              <p className="text-gray-700 font-semibold mt-2">Request Accepted</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-orange-500 text-white p-2 rounded-full">
                <i className="fas fa-check"></i>
              </div>
              <p className="text-gray-700 font-semibold mt-2">Request Verified</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-orange-300 text-white p-2 rounded-full">
                <i className="fas fa-truck"></i>
              </div>
              <p className="text-gray-700 font-semibold mt-2">Picker on the Way</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-gray-300 text-white p-2 rounded-full">
                <i className="fas fa-home"></i>
              </div>
              <p className="text-gray-400 font-semibold mt-2">Reached Doorstep</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-gray-300 text-white p-2 rounded-full">
                <i className="fas fa-recycle"></i>
              </div>
              <p className="text-gray-400 font-semibold mt-2">Picked E-Waste</p>
            </div>
            <div className="flex flex-col items-center">
              <div className="bg-gray-300 text-white p-2 rounded-full">
                <i className="fas fa-check-circle"></i>
              </div>
              <p className="text-gray-400 font-semibold mt-2">Successfully Recycled</p>
            </div>
          </div>
        </div>

        {/* Tracking Timeline */}
        <div className="py-4">
          <h3 className="text-lg font-bold text-gray-800">E-Waste Collection Process</h3>
          <ul className="list-disc list-inside">
            <li className="text-orange-500">Request Made (08/22/2022 15:29)</li>
            <li className="text-orange-500">Request Accepted (08/22/2022 16:00)</li>
            <li className="text-orange-500">Request Verified (08/22/2022 16:30)</li>
            <li className="text-orange-300">Picker on the Way (Expected: 08/23/2022 09:00)</li>
            <li className="text-gray-600">Reached Doorstep (Pending)</li>
            <li className="text-gray-600">Picked E-Waste (Pending)</li>
            <li className="text-gray-600">Successfully Recycled (Pending)</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default EWasteTrackingPage;
  