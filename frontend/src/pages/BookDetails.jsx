import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import axios from 'axios';

export default function BookDetails() {
  const { id } = useParams();
  const [book, setBook] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBookAndRecommendations();
  }, [id]);

  const fetchBookAndRecommendations = async () => {
    try {
      setLoading(true);
      const res = await axios.get(`http://localhost:8000/api/v1/recommend/hybrid/${id}`);
      setBook(res.data.source_book);
      setRecommendations(res.data.recommendations);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return (
    <div className="flex flex-col justify-center items-center py-32 space-y-4">
      <div className="w-16 h-16 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin" />
      <p className="text-indigo-300 font-medium animate-pulse">Running ML prediction pipeline...</p>
    </div>
  );
  
  if (!book) return <div className="text-center py-32 text-gray-400">Book not found in the processed dataset.</div>;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 pt-8">
      {/* Source Book Details */}
      <div className="lg:col-span-4">
        <div className="glass-panel p-8 rounded-3xl sticky top-24 border-t-4 border-t-indigo-500 shadow-[0_0_40px_rgba(99,102,241,0.1)]">
          <div className="mb-6 inline-block rounded-full bg-indigo-500/10 px-3 py-1 text-sm font-semibold text-indigo-400 border border-indigo-500/20">
            Source Book
          </div>
          <h1 className="text-3xl font-extrabold text-white mb-2 leading-tight">{book['Book-Title']}</h1>
          <p className="text-xl text-purple-400 font-medium mb-8">{book['Book-Author']}</p>
          
          <div className="space-y-4 border-t border-white/10 pt-6">
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider font-bold mb-1">Publisher</p>
              <p className="text-gray-300">{book.Publisher}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider font-bold mb-1">Publication Year</p>
              <p className="text-gray-300">{book['Year-Of-Publication']}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider font-bold mb-1">ISBN</p>
              <p className="text-gray-300 font-mono text-sm">{book.ISBN}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Hybrid Recommendations */}
      <div className="lg:col-span-8">
        <div className="flex items-center gap-4 mb-8 pl-2">
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-purple-500 to-pink-500 flex items-center justify-center shadow-lg">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-white">AI Recommendations</h2>
        </div>
        
        <div className="space-y-6">
          {recommendations.map((rec, idx) => (
            <div key={idx} className="glass-panel p-6 rounded-2xl flex flex-col sm:flex-row gap-6 hover:bg-white/10 transition-colors group relative overflow-hidden">
              
              {/* Score indicator */}
              <div className="absolute top-0 right-0 bottom-0 w-1 bg-gradient-to-b from-indigo-500 to-purple-500 opacity-50 group-hover:opacity-100 transition-opacity" />
              
              <div className="flex-1">
                <Link to={`/book/${rec.book.ISBN}`} className="text-xl font-bold text-indigo-300 hover:text-indigo-200 transition-colors">
                  {rec.book['Book-Title']}
                </Link>
                <p className="text-md text-gray-400 mt-1 mb-4">{rec.book['Book-Author']}</p>
                
                <div className="bg-black/20 rounded-xl p-4 border border-white/5">
                  <p className="text-xs font-semibold text-gray-500 uppercase tracking-widest mb-2 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
                    Why it matches
                  </p>
                  <ul className="space-y-1">
                    {rec.reasons.map((r, i) => (
                      <li key={i} className="text-sm text-gray-300 flex items-start gap-2">
                        <span className="text-indigo-500 mt-0.5">•</span> {r}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
              
              <div className="flex flex-col justify-center items-end min-w-[120px] sm:border-l sm:border-white/10 sm:pl-6">
                <div className="text-4xl font-black bg-clip-text text-transparent bg-gradient-to-br from-indigo-400 to-purple-400">
                  {(rec.similarity_score * 100).toFixed(0)}%
                </div>
                <div className="text-xs text-gray-500 uppercase tracking-widest font-bold mt-1">Match</div>
                
                <div className="mt-4 w-full flex flex-col gap-1 text-[10px] text-right text-gray-500">
                  <div>Collab: {(rec.collaborative_score * 100).toFixed(1)}%</div>
                  <div>Content: {(rec.content_score * 100).toFixed(1)}%</div>
                </div>
              </div>
            </div>
          ))}
          
          {recommendations.length === 0 && (
            <div className="glass-panel p-8 rounded-2xl text-center text-gray-400">
              No strong recommendations found for this specific title.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
