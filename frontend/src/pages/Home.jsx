import { Link } from 'react-router-dom';

export default function Home() {
  return (
    <div className="relative isolate overflow-hidden">
      <div className="mx-auto max-w-4xl py-24 sm:py-32 lg:py-40">
        <div className="text-center">
          <div className="mb-8 flex justify-center">
            <div className="relative rounded-full px-4 py-1 text-sm leading-6 text-indigo-300 ring-1 ring-indigo-500/30 hover:ring-indigo-500/50 glass-panel cursor-default transition-all shadow-[0_0_15px_rgba(79,70,229,0.2)]">
              Powered by Collaborative Filtering & TF-IDF Vectors
            </div>
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-white sm:text-7xl mb-6">
            Find Your Next <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
              Favorite Book
            </span>
          </h1>
          <p className="mt-6 text-lg leading-8 text-gray-400 max-w-2xl mx-auto">
            Discover books explicitly tailored to your reading patterns. Our machine learning engine analyzes thousands of books and millions of ratings to find the perfect match.
          </p>
          <div className="mt-10 flex items-center justify-center gap-x-6">
            <Link
              to="/discover"
              className="rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 px-8 py-3.5 text-sm font-semibold text-white shadow-[0_0_20px_rgba(99,102,241,0.4)] hover:shadow-[0_0_30px_rgba(99,102,241,0.6)] transition-all hover:-translate-y-1"
            >
              Start Exploring
            </Link>
            <Link to="/analytics" className="text-sm font-semibold leading-6 text-gray-300 hover:text-white transition-colors group">
              View Dataset Analytics <span aria-hidden="true" className="group-hover:translate-x-1 inline-block transition-transform">→</span>
            </Link>
          </div>
        </div>
      </div>
      
      {/* Decorative blobs */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -z-10 w-[800px] h-[800px] opacity-20 pointer-events-none blur-3xl rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 mix-blend-screen" />
    </div>
  );
}
