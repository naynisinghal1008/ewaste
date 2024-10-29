import React from "react";
import { Link } from "react-router-dom";

const Section1 = () => {
          return (
            <div className="overflow-x-hidden">
              <section className="relative w-screen h-screen flex flex-col justify-center">
                <video
                  autoPlay
                  loop
                  muted
                  className="absolute inset-0 w-full h-full object-cover"
                >
                  <source src={'ewaste_main.mp4'} type="video/mp4" />
                  Your browser does not support the video tag.
                </video>
                <div className="absolute inset-0 bg-black opacity-60"></div>
        
                <div className="relative flex flex-col w-[37%] pl-5 z-10 ml-10">
                  <h1 className="text-6xl font-bold text-white">
                    Welcome to{""}
                    <br></br>
                    <span className="text-orange-700 text-7xl font-playfair">E-WISE</span>
                  </h1>
                  <h3 className="text-xl text-gray-200 mt-4 font-serif">
                  E-WISE addresses the growing e-waste problem by offering convenient, eco-friendly disposal solutions. Users can schedule pickups at their preferred time and location while accessing information on nearby collection centers and educational resources. This approach encourages responsible e-waste management and promotes environmental sustainability
                  </h3>
                </div>
                <div className="relative flex gap-4 pl-5 mt-4 z-10 ml-10">
                  <Link to="/LoginAsAdmin">
                    <button className="border border-richblack-700 bg-richblack-800 text-[12px] lg:text-[16px] px-2 py-1 lg:px-[12px] lg:py-[8px] bg-yellow-50 text-black font-semibold rounded-md hover:scale-95 transition-all duration-200">
                      Admin Login
                    </button>
                  </Link>
        
                  <Link to="/login">
                    <button className="text-orange-700 border border-richblack-700 bg-richblack-700 text-[12px] lg:text-[16px] px-2 py-1 lg:px-[12px] lg:py-[8px] text-richblack-5 font-semibold rounded-md hover:scale-95 transition-all duration-200">
                      Log In
                    </button>
                  </Link>
                </div>
              </section>
        
              <section className="py-12 bg-white-100 flex justify-center items-center">
                <div className="flex space-x-2 flex items-center justify-center ">
                  <div className="flex justify-center items-center w-50 h-40 p-3 bg-orange-700 text-white rounded-full shadow-lg">
                    <p className="text-center text-3xl font-bold px-4">What We Do</p>
                  </div>
                          <div className="flex flex-wrap justify-center items-center w-[70%] h-[40%] text-orange-800 rounded-full shadow-lg border border-orange-300 p-6">
                    <p className="text-lg text-center text-black">
                    Our project utilizes advanced machine learning algorithms and GPS technology to detect, categorize, and report instances of waste, particularly e-waste, in urban areas. By leveraging public vehicle cameras and community involvement, we aim to streamline the waste management process and alert municipal authorities in real-time, ensuring prompt action.
                    </p>
                  </div>
                </div>
              </section>
            </div>
          );
        };
        
        export default Section1;
        

