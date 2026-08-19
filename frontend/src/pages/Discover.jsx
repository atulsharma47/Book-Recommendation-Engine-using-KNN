import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';

export default function Discover() {
  const [query, setQuery] = useState('');
  const [books, setBooks] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDefaultBooks();
  }, []);

  const fetchDefaultBooks = async () => {
    try {
      setLoading(true);
      const res = await axios.get('http://localhost:8000/api/v1/books?limit=24');
      setBooks(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query) return fetchDefaultBooks();
    try {
      setLoading(true);
      const res = await axios.get(`http://localhost:8000/api/v1/search?q=${query}`);
      setBooks(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="py-8">
      <div className="mb-12 max-w-3xl mx-auto text-center">
        <h2 className="text-3xl font-bold text-white mb-6">Discover Our Library</h2>
        <form onSubmit={handleSearch} className="flex gap-4 p-2 glass-panel rounded-2xl shadow-2xl">
          <input
            type="text"
            className="flex-1 bg-transparent border-0 py-3 text-white placeholder:text-gray-400 focus:ring-0 sm:text-lg sm:leading-6 px-6 outline-none"
            placeholder="Search by title, author, or publisher..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            type="submit"
            className="rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 px-6 py-3 text-sm font-semibold text-white shadow-md hover:shadow-lg transition-all hover:scale-105"
          >
            Search
          </button>
        </form>
      </div>

      {loading ? (
        <div className="flex justify-center items-center py-24">
          <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {books.map((book) => (
            <Link 
              key={book.ISBN} 
              to={`/book/${book.ISBN}`}
              className="group glass-panel rounded-2xl p-6 hover:-translate-y-2 hover:shadow-[0_10px_30px_rgba(99,102,241,0.2)] transition-all cursor-pointer relative overflow-hidden flex flex-col h-full"
            >
              <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity">
                <span className="bg-indigo-500/20 text-indigo-300 text-xs px-2 py-1 rounded-full border border-indigo-500/30">
                  {book['Year-Of-Publication']}
                </span>
              </div>
              
              <div className="flex-1">
                <h3 className="text-xl font-bold text-gray-100 group-hover:text-indigo-400 transition-colors line-clamp-2 mb-2">
                  {book['Book-Title']}
                </h3>
                <p className="text-sm font-medium text-purple-400 mb-4">{book['Book-Author']}</p>
              </div>
              
              <div className="mt-auto border-t border-white/10 pt-4">
                <p className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Publisher</p>
                <p className="text-sm text-gray-400 truncate">{book.Publisher}</p>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
