import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import Home from './pages/Home';
import Discover from './pages/Discover';
import BookDetails from './pages/BookDetails';
import Analytics from './pages/Analytics';

function Navigation() {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="sticky top-0 z-50 glass-panel border-b border-white/10 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="flex-shrink-0 flex items-center gap-3 group">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold shadow-lg group-hover:scale-105 transition-transform">
                B
              </div>
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-purple-400 tracking-tight">
                BookWise AI
              </span>
            </Link>
            <div className="hidden sm:ml-10 sm:flex sm:space-x-8">
              <Link 
                to="/discover" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${isActive('/discover') ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'}`}
              >
                Discover
              </Link>
              <Link 
                to="/analytics" 
                className={`inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors ${isActive('/analytics') ? 'border-purple-500 text-purple-400' : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'}`}
              >
                Analytics
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
}

function App() {
  return (
    <Router>
      <div className="min-h-screen text-gray-100 font-sans selection:bg-indigo-500/30">
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/discover" element={<Discover />} />
            <Route path="/book/:id" element={<BookDetails />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
