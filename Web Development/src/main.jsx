import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import { createBrowserRouter, createRoutesFromElements, RouterProvider } from 'react-router-dom'
import Home from './components/Home/Home.jsx'
import About from './components/About/About.jsx'
import App from './App.jsx'
import Contact from './components/Contact/Contact.jsx'
import { Route } from 'react-router-dom'
import Login from './components/Login/Login.jsx'
import RecyclingCenters from './components/recyclingCentres/recyclingCentres.jsx'
import ImageUpload from './components/ImageUpload/ImageUpload.jsx'
import Signup from './components/SignUp/SignUp.jsx'
import Resource from './components/Resource/Resource.jsx'
import LeaderBoard from './components/LeaderBoard/LeaderBoard.jsx'
import Tracker from './components/Tracker/Tracker.jsx'
import LoginAsAdmin from './components/Login/LoginAsAdmin.jsx'
import Admin from './components/Admin/Admin.jsx'



const router = createBrowserRouter(
  createRoutesFromElements(
    <Route path='/' element={<App/>}>
      <Route path='' element={<Home/>}/>
      <Route path='/about' element={<About/>}/>
      <Route path='/contact' element={<Contact/>}/>
      <Route path="/login" element={<Login />} />
      <Route path="/recycling-centers" element={<RecyclingCenters />} />
      <Route path="/ImageUpload" element={<ImageUpload/>}/>
      <Route path="/SignUp" element={<Signup/>}/>
      <Route path="/Resource" element={<Resource/>}/>
      <Route path='/LeaderBoard' element={<LeaderBoard/>}/>
      <Route path='/Tracker' element={<Tracker/>}/>
      <Route path='/LoginAsAdmin' element={<LoginAsAdmin/>}/>
      <Route path='/Admin' element={<Admin/>}/>
    </Route>
  )
)
ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router}/> 
  </React.StrictMode>
)

