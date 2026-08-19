import { useState, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

export default function Analytics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const res = await axios.get('http://localhost:8000/api/v1/analytics');
      setMetrics(res.data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return (
    <div className="flex justify-center items-center py-32">
      <div className="w-12 h-12 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin" />
    </div>
  );
  
  if (!metrics) return <div className="text-center py-12 text-gray-400">Failed to load analytics.</div>;

  const data = [
    { name: 'Books', count: metrics.total_books_processed, color: '#818cf8' },
    { name: 'Users', count: metrics.total_users_processed, color: '#c084fc' },
  ];

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-panel p-3 rounded-lg border-white/20 shadow-xl">
          <p className="text-white font-bold">{`${payload[0].payload.name}: ${payload[0].value.toLocaleString()}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-10 py-8">
      <div className="text-center max-w-2xl mx-auto mb-12">
        <h1 className="text-4xl font-extrabold text-white mb-4 tracking-tight">Dataset Insights</h1>
        <p className="text-lg text-gray-400">Deep dive into the matrix that powers our recommendation engine.</p>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
        <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full blur-2xl -mr-10 -mt-10 group-hover:bg-indigo-500/20 transition-colors" />
          <dt className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Processed Books</dt>
          <dd className="text-5xl font-black text-white">{metrics.total_books_processed.toLocaleString()}</dd>
        </div>
        <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/10 rounded-full blur-2xl -mr-10 -mt-10 group-hover:bg-purple-500/20 transition-colors" />
          <dt className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Active Users</dt>
          <dd className="text-5xl font-black text-white">{metrics.total_users_processed.toLocaleString()}</dd>
        </div>
        <div className="glass-panel rounded-2xl p-6 relative overflow-hidden group border-indigo-500/30">
          <div className="absolute top-0 right-0 w-32 h-32 bg-pink-500/10 rounded-full blur-2xl -mr-10 -mt-10 group-hover:bg-pink-500/20 transition-colors" />
          <dt className="text-sm font-bold text-gray-400 uppercase tracking-wider mb-2">Total Ratings</dt>
          <dd className="text-5xl font-black bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 to-pink-400">
            {metrics.total_ratings_processed.toLocaleString()}
          </dd>
        </div>
      </div>

      <div className="glass-panel p-8 rounded-3xl mt-8">
        <div className="flex justify-between items-end mb-8 border-b border-white/10 pb-4">
          <div>
            <h3 className="text-xl font-bold text-white">Entity Distribution</h3>
            <p className="text-sm text-gray-400 mt-1">Breakdown of the core dataset</p>
          </div>
          <div className="text-right">
            <p className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-1">Matrix Sparsity</p>
            <p className="text-xl font-black text-indigo-400">{(metrics.sparsity * 100).toFixed(4)}%</p>
          </div>
        </div>
        
        <div className="h-80 mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#ffffff15" vertical={false} />
              <XAxis dataKey="name" stroke="#9ca3af" tick={{fill: '#9ca3af'}} axisLine={false} tickLine={false} />
              <YAxis stroke="#9ca3af" tick={{fill: '#9ca3af'}} axisLine={false} tickLine={false} />
              <Tooltip content={<CustomTooltip />} cursor={{fill: '#ffffff05'}} />
              <Bar dataKey="count" radius={[8, 8, 0, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
