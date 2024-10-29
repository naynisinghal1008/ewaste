
import React, { useState, useContext } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { FormContext } from '../FormContext/FormContext';

const ImageUpload = () => {
  const [formData, setFormData] = useState({
    date: '',
    time: '',
    address: '',
    instructions: '',
    wasteType: '',
    image: null,
  });

  const { addFormSubmission } = useContext(FormContext);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleFileChange = (e) => {
    setFormData((prev) => ({
      ...prev,
      image: e.target.files[0],
    }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    addFormSubmission(formData);
    toast.success('Request submitted successfully!', {
      position: 'top-right',
      autoClose: 3000,
      hideProgressBar: false,
      closeOnClick: true,
      pauseOnHover: true,
      draggable: true,
      progress: undefined,
    });
    setFormData({
      date: '',
      time: '',
      address: '',
      instructions: '',
      wasteType: '',
      image: null,
    });
  };

  return (
    <div className="flex justify-center items-center min-h-screen p-6 bg-gray-100">
     <div className="grid grid-cols-2 gap-10 bg-white p-10 rounded-lg shadow-lg">
       <div className="flex flex-col justify-center ">
         <div className="flex justify-center">
           <img
            src={"recycling.webp"} 
            alt="E-Waste Management"
            className="w-3/4 h-auto max-w-md rounded-lg shadow-lg pt-0 mt-[-20px] mb-10 text-10xl"
          ></img>
          </div>
          <h2 className="text-2xl font-semibold text-orange-800 mb-4 text-center">
            What you need to do?
          </h2>
          <p className="text-gray-700">
            Please fill out the form beside to schedule a pickup for your
            e-waste. You need to provide a preferred date, time, address for
            pickup, type of e-waste, and any additional instructions. Ensure to
            upload a picture of the items you want to dispose of.
          </p>
        </div>

        <div className="bg-gray-50 p-6 rounded-lg border border-gray-300 shadow-sm">
          <form onSubmit={handleSubmit} className="flex flex-col space-y-4">
            <div>
              <label className="block text-sm font-medium text-orange-800">
                Date *
              </label>
              <input
                type="date"
                name="date"
                value={formData.date}
                onChange={handleChange}
                required
                className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-orange-800">
                Time *
              </label>
              <input
                type="time"
                name="time"
                value={formData.time}
                onChange={handleChange}
                required
                className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-orange-800">
                Address *
              </label>
              <textarea
                name="address"
                value={formData.address}
                onChange={handleChange}
                required
                className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
                placeholder="Enter your address"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-orange-800">
                Type of E-Waste
              </label>
              <input
                type="text"
                name="wasteType"
                value={formData.wasteType}
                onChange={handleChange}
                className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
                placeholder="E.g., Batteries, Appliances"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-orange-800">
                Additional Instructions
              </label>
              <textarea
                name="instructions"
                value={formData.instructions}
                onChange={handleChange}
                className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
                placeholder="Add any special instructions here"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-orange-800">
                Upload Image *
              </label>
              <input
                type="file"
                name="image"
                accept="image/*"
                onChange={handleFileChange}
                required
                className="mt-1 block w-full p-2 border border-gray-300 rounded-md"
              />
            </div>

            <button
              type="submit"
              className="bg-orange-800 text-white font-semibold py-2 px-4 rounded-md hover:bg-orange-700"
            >
              Submit
            </button>
          </form>
        </div>
      </div>

      <ToastContainer />
    </div>
  );
};

export default ImageUpload;

