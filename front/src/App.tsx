import { lazy, Suspense } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  QueryCache,
  MutationCache,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { toast } from "sonner";
import { ApiError } from "@/lib/api";
import { TopLoadingBar } from "@/components/ui/TopLoadingBar";
import { ErrorBoundary } from "@/components/layout/ErrorBoundary";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { AppBackground } from "@/components/layout/AppBackground";

// Landing page stays eager — first paint is what users see on cold load.
import Index from "./pages/Index";

// Everything else is route-split. Recharts, Framer Motion, shadcn Command,
// and the heavier page components ride along in per-route chunks.
const Dashboard   = lazy(() => import("./pages/Dashboard"));
const Import      = lazy(() => import("./pages/Import"));
const Train       = lazy(() => import("./pages/Train"));
const Predictions = lazy(() => import("./pages/Predictions"));
const Leaderboard = lazy(() => import("./pages/Leaderboard"));
const Analysis    = lazy(() => import("./pages/Analysis"));
const Verify      = lazy(() => import("./pages/Verify"));
const About       = lazy(() => import("./pages/About"));
const NotFound    = lazy(() => import("./pages/NotFound"));

const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error) => {
      if (error instanceof ApiError) {
        toast.error(`API Error ${error.status}: ${error.detail}`);
      } else {
        toast.error("An unexpected error occurred");
      }
    },
  }),
  mutationCache: new MutationCache({
    onError: (error) => {
      if (error instanceof ApiError) {
        toast.error(`API Error ${error.status}: ${error.detail}`);
      } else {
        toast.error("An unexpected error occurred");
      }
    },
  }),
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30_000,
    },
  },
});

// Shown while a lazy route's chunk downloads. Sits BELOW the Navbar so the
// menu stays clickable during the swap — otherwise users perceive their click
// as "didn't work" and click again.
function RouteFallback() {
  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center bg-background pt-24">
      <div className="h-8 w-8 animate-spin rounded-full border-2 border-orange-500/30 border-t-orange-500" />
    </div>
  );
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TopLoadingBar />
    <TooltipProvider>
      <AppBackground />
      <Toaster />
      <Sonner />
      <BrowserRouter>
        {/* Navbar + Footer live OUTSIDE ErrorBoundary so a page render error
            still leaves the chrome usable (user can navigate to recover).
            Outside Suspense too, so they don't unmount during lazy-route loads. */}
        <Navbar />
        <ErrorBoundary>
          <Suspense fallback={<RouteFallback />}>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/import" element={<Import />} />
              <Route path="/train" element={<Train />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/leaderboard" element={<Leaderboard />} />
              <Route path="/analysis" element={<Analysis />} />
              <Route path="/verify" element={<Verify />} />
              <Route path="/about" element={<About />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
        <Footer />
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
