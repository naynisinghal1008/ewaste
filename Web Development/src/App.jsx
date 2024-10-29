import './App.css'
import Header from './components/Header/Header'
import Footer from './components/Footer/Footer'
import { Outlet } from 'react-router-dom'
import { FormProvider } from './components/FormContext/FormContext'


function App() {
  
  return (
    <>   
    <Header/>
    <FormProvider>
    <Outlet/>
    </FormProvider>
    <Footer/>
    
    </>
  )
}

export default App


/*

Yes, that's  true. In React applications, including those using React Router for navigation, the `<Link>` tag provided by React Router is preferred over the traditional `<a>` tag for several reasons related to how React manages rendering and state.

	1. Preventing Full Page Reloads : When you use an `<a>` tag with an `href` attribute, clicking on it typically triggers a full page reload, especially if the `href` points to a different route or location. This behavior is contrary to React's single-page application (SPA) model, where the page should not reload entirely during navigation but rather update dynamically.

	2. Client-Side Routing : React Router provides client-side routing, which means it intercepts link clicks and handles navigation without requesting new pages from the server. This results in a smoother user experience as only the necessary components are re-rendered based on the URL change.

	3. Maintaining State : When you navigate using `<Link>` from React Router, React maintains the current state of components unless explicitly instructed otherwise. This is crucial for preserving application state and context across different views or pages.

	4. Performance : Since React Router manages routing on the client side, it typically results in faster navigation between pages compared to traditional server-side navigation where a full page reload is required.

	5. Integration with React's Virtual DOM : React's virtual DOM allows it to efficiently update only the parts of the page that have changed. React Router's `<Link>` component integrates seamlessly with this paradigm, ensuring that only the necessary components are re-rendered when navigating between routes.

In summary, using `<Link>` from React Router instead of `<a>` tags helps adhere to React's principles of efficient rendering and state management in SPAs. It ensures that navigation is fast, smooth, and maintains the integrity of React's virtual DOM for optimal performance.
*/