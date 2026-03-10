import { useState } from "react";
import { motion } from "framer-motion";
import { Calendar, Filter, RefreshCw, Download } from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { PredictionCard } from "@/components/shared/PredictionCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const predictions = [
  {
    homeTeam: "Lakers",
    awayTeam: "Celtics",
    predictedWinner: "home" as const,
    confidence: 72,
    date: "Jan 25, 2024",
    model: "XGBoost",
  },
  {
    homeTeam: "Warriors",
    awayTeam: "Heat",
    predictedWinner: "home" as const,
    confidence: 68,
    date: "Jan 25, 2024",
    model: "Random Forest",
  },
  {
    homeTeam: "Bucks",
    awayTeam: "76ers",
    predictedWinner: "away" as const,
    confidence: 55,
    date: "Jan 26, 2024",
    model: "XGBoost",
  },
  {
    homeTeam: "Nuggets",
    awayTeam: "Suns",
    predictedWinner: "home" as const,
    confidence: 81,
    date: "Jan 26, 2024",
    model: "XGBoost",
  },
  {
    homeTeam: "Mavericks",
    awayTeam: "Thunder",
    predictedWinner: "away" as const,
    confidence: 62,
    date: "Jan 27, 2024",
    model: "Neural Network",
  },
  {
    homeTeam: "Clippers",
    awayTeam: "Knicks",
    predictedWinner: "home" as const,
    confidence: 74,
    date: "Jan 27, 2024",
    model: "Random Forest",
  },
];

export default function Predictions() {
  const [searchTerm, setSearchTerm] = useState("");

  const filteredPredictions = predictions.filter(
    (p) =>
      p.homeTeam.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.awayTeam.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 pt-24 pb-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <h1 className="text-3xl font-bold md:text-4xl">Predictions</h1>
          <p className="mt-2 text-muted-foreground">
            View upcoming game predictions and confidence scores
          </p>
        </motion.div>

        {/* Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex flex-1 items-center gap-4">
            <div className="relative flex-1 max-w-md">
              <Input
                placeholder="Search teams..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="bg-muted/30 border-border/50 pl-10"
              />
              <Filter className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            </div>
            <Button variant="outline" size="icon">
              <Calendar className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" className="gap-2">
              <RefreshCw className="h-4 w-4" />
              Refresh
            </Button>
            <Button variant="outline" className="gap-2">
              <Download className="h-4 w-4" />
              Export
            </Button>
          </div>
        </motion.div>

        {/* Predictions Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {filteredPredictions.map((prediction, index) => (
            <PredictionCard key={index} {...prediction} delay={0.2 + index * 0.1} />
          ))}
        </div>

        {filteredPredictions.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="py-20 text-center"
          >
            <p className="text-muted-foreground">No predictions found matching your search</p>
          </motion.div>
        )}

        {/* Confidence Legend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="mt-12"
        >
          <div className="glass-card p-6">
            <h3 className="mb-4 font-semibold">Confidence Levels</h3>
            <div className="flex flex-wrap gap-6">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-success" />
                <span className="text-sm text-muted-foreground">
                  High (&gt;70%)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-warning" />
                <span className="text-sm text-muted-foreground">
                  Medium (50-70%)
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded-full bg-destructive" />
                <span className="text-sm text-muted-foreground">
                  Low (&lt;50%)
                </span>
              </div>
            </div>
          </div>
        </motion.div>
      </main>

      <Footer />
    </div>
  );
}
