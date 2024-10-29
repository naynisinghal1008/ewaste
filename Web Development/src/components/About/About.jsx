


import React from 'react';
import './about.css'

export default function About() {
  return (
    <div className="py-16 bg-gray-50 opacity-0 transition-opacity duration-1000 ease-in-out animate-fade-in">
    <div className="container mx-auto px-6 text-gray-800 md:px-12 xl:px-6">
    <div className="flex items-center space-y-8 md:space-y-0 md:flex-row md:gap-8 lg:gap-12">
          <div className="md:w-5/12 lg:w-5/12 flex justify-center">
            <img
              src="https://tailus.io/sources/blocks/left-image/preview/images/startup.png"
              alt="Waste Management Illustration"
              className="rounded-lg shadow-lg"
            />
          </div>
          <div className="md:w-7/12 lg:w-6/12">
            <h2 className="text-3xl font-extrabold text-gray-900 md:text-4xl">
              Our Mission
            </h2>
            <p className="mt-4 text-lg text-gray-700">
              At EWISE, our mission is to create a cleaner, more sustainable environment by effectively managing electronic waste (e-waste) through innovative technology. We believe that every community deserves a clean and safe environment, and we are committed to making that a reality.
            </p>
            <h2 className="mt-8 text-3xl font-extrabold text-gray-900 md:text-4xl">
              What We Do
            </h2>
            <p className="mt-4 text-lg text-gray-700">
              Our project utilizes advanced machine learning algorithms and GPS technology to detect, categorize, and report instances of waste, particularly e-waste, in urban areas. By leveraging public vehicle cameras and community involvement, we aim to streamline the waste management process and alert municipal authorities in real-time, ensuring prompt action.
            </p>
            <h2 className="mt-8 text-3xl font-extrabold text-gray-900 md:text-4xl">
              How It Works
            </h2>
            <p className="mt-4 text-lg text-gray-700">
              <strong>Waste Image Capturing:</strong> Equipped public vehicles capture images of waste during their daily routes. These images are then pre-filtered using machine learning to ensure quality and relevance.<br />
              <strong>Waste Detection:</strong> Our sophisticated ML model identifies and categorizes waste types, tagging them with precise GPS locations for accurate reporting.<br />
              <strong>Real-Time Alerts:</strong> When waste is detected, alerts are sent to municipal authorities, allowing them to respond swiftly and efficiently.<br />
              <strong>Community Engagement:</strong> We empower citizens to participate by allowing them to report waste via our mobile app, contributing to a collective effort in maintaining a clean environment.
            </p>
            <h2 className="mt-8 text-3xl font-extrabold text-gray-900 md:text-4xl">
              Why E-Waste Matters
            </h2>
            <p className="mt-4 text-lg text-gray-700">
              Electronic waste is one of the fastest-growing waste streams globally, and improper disposal can lead to severe environmental and health issues. Our initiative aims to tackle this challenge by promoting responsible disposal and recycling, ensuring that harmful materials do not end up in landfills.
            </p>
          </div>
        </div>
    </div>
  </div>
  );
}
