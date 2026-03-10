import { motion } from "framer-motion";
import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle,
  FileUp,
  TrendingUp,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { MetricCard } from "@/components/shared/MetricCard";
import { NavigationCard } from "@/components/shared/NavigationCard";

const metrics = [
  {
    title: "Total Predictions",
    value: "2,847",
    subtitle: "Last 30 days",
    icon: TrendingUp,
    trend: { value: 12.5, isPositive: true },
  },
  {
    title: "Model Accuracy",
    value: "78.4%",
    subtitle: "Average across all models",
    icon: BarChart3,
    trend: { value: 2.3, isPositive: true },
  },
  {
    title: "Active Models",
    value: "4",
    subtitle: "Ready for predictions",
    icon: Brain,
  },
  {
    title: "Datasets Imported",
    value: "12",
    subtitle: "6,432 total games",
    icon: Activity,
  },
];

const navigationCards = [
  {
    title: "Import Data",
    description: "Upload CSV or Excel files with historical NBA game data",
    href: "/import",
    icon: FileUp,
  },
  {
    title: "Train Models",
    description: "Select and train machine learning models on your data",
    href: "/train",
    icon: Brain,
  },
  {
    title: "View Predictions",
    description: "See upcoming game predictions and confidence scores",
    href: "/predictions",
    icon: TrendingUp,
  },
  {
    title: "Analysis",
    description: "Compare model performance and analyze results",
    href: "/analysis",
    icon: BarChart3,
  },
  {
    title: "Verification",
    description: "Compare predictions with actual game results",
    href: "/verify",
    icon: CheckCircle,
  },
];

export default function Dashboard() {
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
          <h1 className="text-3xl font-bold md:text-4xl">Dashboard</h1>
          <p className="mt-2 text-muted-foreground">
            Overview of your NBA prediction system
          </p>
        </motion.div>

        {/* Metrics Grid */}
        <div className="mb-12 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {metrics.map((metric, index) => (
            <MetricCard key={metric.title} {...metric} delay={index * 0.1} />
          ))}
        </div>

        {/* Navigation Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="mb-8"
        >
          <h2 className="text-xl font-semibold">Quick Actions</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Navigate to different sections of the application
          </p>
        </motion.div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {navigationCards.map((card, index) => (
            <NavigationCard key={card.title} {...card} delay={0.5 + index * 0.1} />
          ))}
        </div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1 }}
          className="mt-12"
        >
          <h2 className="mb-6 text-xl font-semibold">Recent Activity</h2>
          <div className="glass-card overflow-hidden">
            <div className="divide-y divide-border/50">
              {[
                {
                  action: "Model trained",
                  detail: "XGBoost on 2024 Season data",
                  time: "2 hours ago",
                  status: "success",
                },
                {
                  action: "Data imported",
                  detail: "nba_games_2024.csv (1,230 games)",
                  time: "5 hours ago",
                  status: "success",
                },
                {
                  action: "Predictions generated",
                  detail: "15 upcoming games",
                  time: "1 day ago",
                  status: "info",
                },
                {
                  action: "Verification complete",
                  detail: "87% accuracy on last batch",
                  time: "2 days ago",
                  status: "success",
                },
              ].map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 1.1 + index * 0.1 }}
                  className="flex items-center justify-between px-6 py-4"
                >
                  <div className="flex items-center gap-4">
                    <div
                      className={`h-2 w-2 rounded-full ${
                        item.status === "success" ? "bg-success" : "bg-secondary"
                      }`}
                    />
                    <div>
                      <p className="font-medium">{item.action}</p>
                      <p className="text-sm text-muted-foreground">{item.detail}</p>
                    </div>
                  </div>
                  <span className="text-sm text-muted-foreground">{item.time}</span>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>
      </main>

      <Footer />
    </div>
  );
}
