import { useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle,
  XCircle,
  Clock,
  Upload,
  Download,
  Filter,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { MetricCard } from "@/components/shared/MetricCard";
import { DropZone } from "@/components/shared/DropZone";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const verificationResults = [
  {
    id: 1,
    homeTeam: "Lakers",
    awayTeam: "Celtics",
    predicted: "Lakers",
    actual: "Lakers",
    status: "correct",
    confidence: 72,
    date: "Jan 20, 2024",
  },
  {
    id: 2,
    homeTeam: "Warriors",
    awayTeam: "Heat",
    predicted: "Warriors",
    actual: "Heat",
    status: "incorrect",
    confidence: 58,
    date: "Jan 20, 2024",
  },
  {
    id: 3,
    homeTeam: "Bucks",
    awayTeam: "76ers",
    predicted: "76ers",
    actual: "76ers",
    status: "correct",
    confidence: 65,
    date: "Jan 21, 2024",
  },
  {
    id: 4,
    homeTeam: "Nuggets",
    awayTeam: "Suns",
    predicted: "Nuggets",
    actual: "Nuggets",
    status: "correct",
    confidence: 81,
    date: "Jan 21, 2024",
  },
  {
    id: 5,
    homeTeam: "Mavericks",
    awayTeam: "Thunder",
    predicted: "Thunder",
    actual: null,
    status: "pending",
    confidence: 62,
    date: "Jan 22, 2024",
  },
];

export default function Verify() {
  const [showUpload, setShowUpload] = useState(false);

  const correctCount = verificationResults.filter((r) => r.status === "correct").length;
  const incorrectCount = verificationResults.filter((r) => r.status === "incorrect").length;
  const pendingCount = verificationResults.filter((r) => r.status === "pending").length;
  const accuracy = (correctCount / (correctCount + incorrectCount)) * 100;

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
          <h1 className="text-3xl font-bold md:text-4xl">Verification</h1>
          <p className="mt-2 text-muted-foreground">
            Compare predictions with actual game results
          </p>
        </motion.div>

        {/* Metrics */}
        <div className="mb-8 grid gap-4 md:grid-cols-4">
          <MetricCard
            title="Accuracy"
            value={`${accuracy.toFixed(1)}%`}
            subtitle="Based on verified results"
            icon={CheckCircle}
            delay={0.1}
          />
          <MetricCard
            title="Correct"
            value={correctCount.toString()}
            subtitle="Predictions matched"
            icon={CheckCircle}
            delay={0.2}
          />
          <MetricCard
            title="Incorrect"
            value={incorrectCount.toString()}
            subtitle="Predictions missed"
            icon={XCircle}
            delay={0.3}
          />
          <MetricCard
            title="Pending"
            value={pendingCount.toString()}
            subtitle="Awaiting results"
            icon={Clock}
            delay={0.4}
          />
        </div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex items-center gap-2">
            <Button variant="outline" className="gap-2">
              <Filter className="h-4 w-4" />
              Filter
            </Button>
            <Button variant="outline" className="gap-2">
              <Download className="h-4 w-4" />
              Export Report
            </Button>
          </div>
          <Button
            variant="hero"
            className="gap-2"
            onClick={() => setShowUpload(!showUpload)}
          >
            <Upload className="h-4 w-4" />
            Upload Results
          </Button>
        </motion.div>

        {/* Upload Section */}
        {showUpload && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mb-8"
          >
            <div className="glass-card p-6">
              <h3 className="mb-4 font-semibold">Upload Actual Results</h3>
              <DropZone
                onFileSelect={(file) => console.log("Selected:", file)}
              />
            </div>
          </motion.div>
        )}

        {/* Results Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="glass-card overflow-hidden"
        >
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-border/50 bg-muted/30">
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    Match
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    Date
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    Predicted
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    Actual
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    Confidence
                  </th>
                  <th className="px-6 py-4 text-left text-sm font-semibold">
                    Status
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {verificationResults.map((result, index) => (
                  <motion.tr
                    key={result.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.7 + index * 0.1 }}
                    className="hover:bg-muted/30 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <span className="font-medium">
                        {result.homeTeam} vs {result.awayTeam}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-muted-foreground">
                      {result.date}
                    </td>
                    <td className="px-6 py-4">
                      <span className="font-medium">{result.predicted}</span>
                    </td>
                    <td className="px-6 py-4">
                      {result.actual ? (
                        <span className="font-medium">{result.actual}</span>
                      ) : (
                        <span className="text-muted-foreground">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4">
                      <span className="stats-text">{result.confidence}%</span>
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium",
                          result.status === "correct" &&
                            "bg-success/10 text-success",
                          result.status === "incorrect" &&
                            "bg-destructive/10 text-destructive",
                          result.status === "pending" &&
                            "bg-warning/10 text-warning"
                        )}
                      >
                        {result.status === "correct" && (
                          <CheckCircle className="h-3 w-3" />
                        )}
                        {result.status === "incorrect" && (
                          <XCircle className="h-3 w-3" />
                        )}
                        {result.status === "pending" && (
                          <Clock className="h-3 w-3" />
                        )}
                        {result.status.charAt(0).toUpperCase() +
                          result.status.slice(1)}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </main>

      <Footer />
    </div>
  );
}
