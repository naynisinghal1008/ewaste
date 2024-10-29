
// import React, { createContext, useState } from 'react';

// export const FormContext = createContext();

// export const FormProvider = ({ children }) => {
//   const [formSubmissions, setFormSubmissions] = useState([]);

//   const addFormSubmission = (formData) => {
//     setFormSubmissions((prevSubmissions) => [...prevSubmissions, formData]);
//   };

//   return (
//     <FormContext.Provider value={{ formSubmissions, addFormSubmission }}>
//       {children}
//     </FormContext.Provider>
//   );
// };

import React, { createContext, useState } from 'react';

export const FormContext = createContext();

export const FormProvider = ({ children }) => {
  const [formSubmissions, setFormSubmissions] = useState([]);

  const addFormSubmission = (submission) => {
    setFormSubmissions((prev) => [...prev, submission]);
  };

  return (
    <FormContext.Provider value={{ formSubmissions, addFormSubmission, setFormSubmissions }}>
      {children}
    </FormContext.Provider>
  );
};

