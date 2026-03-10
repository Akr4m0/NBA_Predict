import { motion } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
} from "recharts";
import { Award, BarChart3, TrendingUp } from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { MetricCard } from "@/components/shared/MetricCard";

const modelComparison = [
  { name: "Decision Tree", accuracy: 72, precision: 70, recall: 74, f1: 72 },
  { name: "Random Forest", accuracy: 76, precision: 75, recall: 77, f1: 76 },
  { name: "XGBoost", accuracy: 78, precision: 79, recall: 77, f1: 78 },
  { name: "Neural Network", accuracy: 74, precision: 73, recall: 75, f1: 74 },
];

const confusionData = [
  { name: "True Positive", value: 432 },
  { name: "True Negative", value: 398 },
  { name: "False Positive", value: 87 },
  { name: "False Negative", value: 103 },
];

const trendData = [
  { month: "Oct", accuracy: 71 },
  { month: "Nov", accuracy: 73 },
  { month: "Dec", accuracy: 75 },
  { month: "Jan", accuracy: 78 },
  { month: "Feb", accuracy: 77 },
  { month: "Mar", accuracy: 79 },
];

const featureImportance = [
  { feature: "Home Win %", importance: 0.23 },
  { feature: "Recent Form", importance: 0.19 },
  { feature: "Head-to-Head", importance: 0.15 },
  { feature: "Point Diff", importance: 0.14 },
  { feature: "Rest Days", importance: 0.12 },
  { feature: "Away Win %", importance: 0.10 },
  { feature: "FG %", importance: 0.07 },
];

const COLORS = ["hsl(158, 64%, 52%)", "hsl(217, 91%, 60%)", "hsl(0, 72%, 51%)", "hsl(38, 92%, 50%)"];

export default function Analysis() {
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
          <h1 className="text-3xl font-bold md:text-4xl">Analysis</h1>
          <p className="mt-2 text-muted-foreground">
            Compare model performance and analyze prediction metrics
          </p>
        </motion.div>

        {/* Best Model Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8 glass-card p-6 flex items-center gap-4"
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
            <Award className="h-7 w-7 text-primary" />
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Best Performing Model</p>
            <p className="text-2xl font-bold">
              XGBoost <span className="text-success stats-text">78.4% accuracy</span>
            </p>
          </div>
        </motion.div>

        {/* Metrics */}
        <div className="mb-8 grid gap-4 md:grid-cols-3">
          <MetricCard
            title="Average Accuracy"
            value="75.2%"
            subtitle="Across all models"
            icon={TrendingUp}
            delay={0.2}
          />
          <MetricCard
            title="Total Predictions"
            value="1,020"
            subtitle="Last 30 days"
            icon={BarChart3}
            delay={0.3}
          />
          <MetricCard
            title="Best F1 Score"
            value="0.78"
            subtitle="XGBoost model"
            icon={Award}
            delay={0.4}
          />
        </div>

        {/* Charts Grid */}
        <div className="grid gap-8 lg:grid-cols-2">
          {/* Model Comparison */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="glass-card p-6"
          >
            <h3 className="mb-6 font-semibold">Model Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 20%, 18%)" />
                <XAxis dataKey="name" stroke="hsl(220, 9%, 60%)" fontSize={12} />
                <YAxis stroke="hsl(220, 9%, 60%)" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(222, 40%, 8%)",
                    border: "1px solid hsl(220, 20%, 18%)",
                    borderRadius: "8px",
                  }}
                />
                <Bar dataKey="accuracy" fill="hsl(24, 100%, 50%)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Accuracy Trend */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="glass-card p-6"
          >
            <h3 className="mb-6 font-semibold">Accuracy Trend</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 20%, 18%)" />
                <XAxis dataKey="month" stroke="hsl(220, 9%, 60%)" fontSize={12} />
                <YAxis stroke="hsl(220, 9%, 60%)" fontSize={12} domain={[65, 85]} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(222, 40%, 8%)",
                    border: "1px solid hsl(220, 20%, 18%)",
                    borderRadius: "8px",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="hsl(24, 100%, 50%)"
                  strokeWidth={3}
                  dot={{ fill: "hsl(24, 100%, 50%)", strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Confusion Matrix */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.7 }}
            className="glass-card p-6"
          >
            <h3 className="mb-6 font-semibold">Prediction Breakdown</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={confusionData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {confusionData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(222, 40%, 8%)",
                    border: "1px solid hsl(220, 20%, 18%)",
                    borderRadius: "8px",
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {confusionData.map((item, index) => (
                <div key={item.name} className="flex items-center gap-2 text-sm">
                  <div
                    className="h-3 w-3 rounded-full"
                    style={{ backgroundColor: COLORS[index] }}
                  />
                  <span className="text-muted-foreground">{item.name}</span>
                  <span className="ml-auto font-medium stats-text">{item.value}</span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Feature Importance */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.8 }}
            className="glass-card p-6"
          >
            <h3 className="mb-6 font-semibold">Feature Importance</h3>
            <div className="space-y-4">
              {featureImportance.map((item, index) => (
                <motion.div
                  key={item.feature}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.9 + index * 0.1 }}
                  className="space-y-1"
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">{item.feature}</span>
                    <span className="font-medium stats-text">
                      {(item.importance * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-muted">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${item.importance * 100}%` }}
                      transition={{ duration: 0.8, delay: 1 + index * 0.1 }}
                      className="h-full rounded-full bg-primary"
                    />
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
