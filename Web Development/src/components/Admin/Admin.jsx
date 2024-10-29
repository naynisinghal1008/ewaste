// import React, { useContext } from 'react';
// import { FormContext } from '../FormContext/FormContext';
// const Admin = () => {
//   const { formSubmissions } = useContext(FormContext);

//   return (
//     <div className="p-6 bg-gray-100 min-h-screen">
//       <h1 className="text-2xl font-semibold text-center mb-6">Admin Page</h1>
//       <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
//         {formSubmissions.map((submission, index) => (
//           <div key={index} className="bg-white p-4 rounded-lg shadow-md">
//             <p>Date: {submission.date}</p>
//             <p>Time: {submission.time}</p>
//             <p>Address: {submission.address}</p>
//             <p>Waste Type: {submission.wasteType}</p>
//             <p>Instructions: {submission.instructions}</p>
//             {submission.image && (
//               <img
//                 src={URL.createObjectURL(submission.image)}
//                 alt="Uploaded E-Waste"
//                 className="mt-2"
//               />
//             )}
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default Admin;


import React, { useContext, useState } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import { FormContext } from '../FormContext/FormContext';
import 'react-toastify/dist/ReactToastify.css';

const Admin = () => {
  const { formSubmissions, setFormSubmissions } = useContext(FormContext);
  const [checkedSubmissions, setCheckedSubmissions] = useState([]);

  const handleCheckboxChange = (index) => {
    const newCheckedSubmissions = [...checkedSubmissions];
    if (newCheckedSubmissions.includes(index)) {
      // Uncheck (add it back to the list)
      newCheckedSubmissions.splice(newCheckedSubmissions.indexOf(index), 1);
    } else {
      // Check (remove the card)
      newCheckedSubmissions.push(index);
      toast.success('Request completed successfully!', {
        position: 'top-right',
        autoClose: 3000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
      });
    }
    setCheckedSubmissions(newCheckedSubmissions);
    const updatedFormSubmissions = formSubmissions.filter(
      (_, i) => !newCheckedSubmissions.includes(i)
    );
    setFormSubmissions(updatedFormSubmissions);
  };

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold text-center mb-6 text-orange-800">Admin Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {formSubmissions.map((submission, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-md relative">
            <div className="flex justify-between items-start">
              <p className="text-gray-700">Date: {submission.date}</p>
              <p className="text-gray-700">Time: {submission.time}</p>
            </div>
            <p className="text-gray-700 mb-2">Address: {submission.address}</p>
            <p className="text-gray-700">Waste Type: {submission.wasteType}</p>
            <p className="text-gray-700 mb-2">Instructions: {submission.instructions}</p>
            {submission.image && (
              <img
                src={URL.createObjectURL(submission.image)}
                alt="Uploaded E-Waste"
                className="mt-2 w-full h-40 object-cover rounded-lg"
              />
            )}

            <div className="flex items-center mt-4">
              <input
                type="checkbox"
                onChange={() => handleCheckboxChange(index)}
                className="mr-2"
              />
              <label className="text-sm text-gray-600">Mark as completed</label>
            </div>
          </div>
        ))}
      </div>
      <ToastContainer />
    </div>
  );
};

export default Admin;
