import { Routes, Route, Link, useLocation } from "react-router-dom";
import { BookOpen, Search, Network, Info } from "lucide-react";
import SearchPage from "./pages/SearchPage";
import SuggestPage from "./pages/SuggestPage";
import GraphPage from "./pages/GraphPage";
import AboutPage from "./pages/AboutPage";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

function Layout({ children }: { children: React.ReactNode }) {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Search", icon: <Search size={18} /> },
    { path: "/suggest", label: "Copilot", icon: <BookOpen size={18} /> },
    { path: "/graph", label: "Graph", icon: <Network size={18} /> },
    { path: "/about", label: "About", icon: <Info size={18} /> },
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 flex flex-col font-sans">
      <header className="sticky top-0 z-50 glass border-b border-white/10">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="bg-blue-600 p-1.5 rounded-lg">
              <Network size={20} className="text-white" />
            </div>
            <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
              Research Copilot GNN
            </span>
          </div>
          
          <nav className="flex items-center gap-1">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
                  location.pathname === item.path
                    ? "bg-white/10 text-white"
                    : "text-zinc-400 hover:text-zinc-100 hover:bg-white/5"
                )}
              >
                {item.icon}
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1 flex flex-col">{children}</main>
    </div>
  );
}

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<SearchPage />} />
        <Route path="/suggest" element={<SuggestPage />} />
        <Route path="/graph" element={<GraphPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </Layout>
  );
}

export default App;
