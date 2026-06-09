import { motion } from "framer-motion";
import {
  BarChart3,
  Brain,
  CheckCircle,
  FileUp,
  TrendingUp,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import { NavigationCard } from "@/components/shared/NavigationCard";
import { Skeleton } from "@/components/ui/skeleton";
import { GlassCard } from "@/components/ui/GlassCard";
import { EmptyState } from "@/components/ui/EmptyState";
import { getResults, getModels, getImports } from "@/lib/api";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(value: number | null | undefined, decimals = 1): string {
  if (value == null) return "—";
  return (value * 100).toFixed(decimals) + "%";
}

// ── KPI Card ──────────────────────────────────────────────────────────────

function KpiCard({
  label,
  value,
  loading,
}: {
  label: string;
  value: string | number;
  loading: boolean;
}) {
  return (
    <GlassCard>
      {loading ? (
        <>
          <Skeleton className="mb-3 h-4 w-24" />
          <Skeleton className="h-8 w-16" />
        </>
      ) : (
        <>
          <p className="mb-1 text-sm text-muted-foreground">{label}</p>
          <p className="text-3xl font-bold">{value}</p>
        </>
      )}
    </GlassCard>
  );
}

// ── Navigation cards ──────────────────────────────────────────────────────

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

// ── Page ──────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const { data: results = [], isLoading: resultsLoading } = useQuery({
    queryKey: ["results"],
    queryFn: () => getResults(),
  });

  const { data: models = [], isLoading: modelsLoading } = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
  });

  const { data: imports = [], isLoading: importsLoading } = useQuery({
    queryKey: ["imports"],
    queryFn: getImports,
  });

  const loading = resultsLoading || modelsLoading || importsLoading;

  // KPI values
  const bestAccuracy = results.length
    ? Math.max(...results.map((r) => r.accuracy ?? 0))
    : null;
  const bestF1 = results.length
    ? Math.max(...results.map((r) => r.f1_score ?? 0))
    : null;

  // Grouped bar chart data
  const barData = results.map((r) => ({
    name: r.name,
    Accuracy: r.accuracy ?? 0,
    Precision: r.precision_score ?? 0,
    Recall: r.recall ?? 0,
    F1: r.f1_score ?? 0,
  }));

  // Line chart — best accuracy per import session date
  const lineData = imports
    .slice()
    .sort(
      (a, b) =>
        new Date(a.import_date).getTime() - new Date(b.import_date).getTime()
    )
    .map((imp) => {
      const sessionResults = results.filter((r) =>
        r.created_date?.startsWith(imp.import_date?.slice(0, 10) ?? "")
      );
      const best = sessionResults.length
        ? Math.max(...sessionResults.map((r) => r.accuracy ?? 0))
        : null;
      return {
        date: imp.import_date?.slice(0, 10) ?? imp.import_date,
        accuracy: best != null ? parseFloat((best * 100).toFixed(2)) : null,
      };
    })
    .filter((d) => d.accuracy != null);

  return (
    <div className="min-h-screen">

      <main className="container mx-auto px-4 pb-12 pt-24">
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

        {/* KPI Row */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-8 grid gap-4 md:grid-cols-2 lg:grid-cols-4"
        >
          <KpiCard
            label="Best Accuracy"
            value={bestAccuracy != null ? fmt(bestAccuracy) : "—"}
            loading={loading}
          />
          <KpiCard
            label="Best F1-Score"
            value={bestF1 != null ? fmt(bestF1) : "—"}
            loading={loading}
          />
          <KpiCard
            label="Total Models"
            value={loading ? "—" : models.length}
            loading={loading}
          />
          <KpiCard
            label="Total Imports"
            value={loading ? "—" : imports.length}
            loading={loading}
          />
        </motion.div>

        {/* Grouped Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-8"
        >
          <GlassCard>
            <h2 className="mb-4 text-lg font-semibold">Model Performance</h2>
            {resultsLoading ? (
              <Skeleton className="h-64 w-full" />
            ) : results.length === 0 ? (
              <EmptyState message="No model results yet. Train a model to see performance here." />
            ) : (
              <ResponsiveContainer width="100%" height={280}>
                <BarChart
                  data={barData}
                  margin={{ top: 4, right: 16, bottom: 4, left: 0 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.07)"
                  />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 12 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 1]}
                  />
                  <Tooltip
                    formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                    contentStyle={{
                      background: "rgba(10,10,20,0.9)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend
                    wrapperStyle={{
                      fontSize: 12,
                      color: "rgba(255,255,255,0.6)",
                    }}
                  />
                  <Bar
                    dataKey="Accuracy"
                    fill="#ff6b00"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="Precision"
                    fill="#ff8533"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar
                    dataKey="Recall"
                    fill="#ffa366"
                    radius={[4, 4, 0, 0]}
                  />
                  <Bar dataKey="F1" fill="#ffbe99" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </GlassCard>
        </motion.div>

        {/* Line Chart — accuracy over import sessions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mb-12"
        >
          <GlassCard>
            <h2 className="mb-4 text-lg font-semibold">Accuracy Over Time</h2>
            {importsLoading || resultsLoading ? (
              <Skeleton className="h-48 w-full" />
            ) : lineData.length === 0 ? (
              <EmptyState message="No session data yet. Import data and train models to track accuracy." />
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart
                  data={lineData}
                  margin={{ top: 4, right: 16, bottom: 4, left: 0 }}
                >
                  <defs>
                    <linearGradient
                      id="accuracyGradient"
                      x1="0"
                      y1="0"
                      x2="0"
                      y2="1"
                    >
                      <stop
                        offset="5%"
                        stopColor="#ff6b00"
                        stopOpacity={0.3}
                      />
                      <stop
                        offset="95%"
                        stopColor="#ff6b00"
                        stopOpacity={0}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.07)"
                  />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tickFormatter={(v) => `${v}%`}
                    tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                    axisLine={false}
                    tickLine={false}
                    domain={[0, 100]}
                  />
                  <Tooltip
                    formatter={(v: number) => [`${v}%`, "Best Accuracy"]}
                    contentStyle={{
                      background: "rgba(10,10,20,0.9)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 8,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#ff6b00"
                    strokeWidth={2}
                    fill="url(#accuracyGradient)"
                    dot={{ fill: "#ff6b00", r: 4 }}
                    activeDot={{ r: 6 }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </GlassCard>
        </motion.div>

        {/* Quick Actions */}
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
            <NavigationCard
              key={card.title}
              {...card}
              delay={0.5 + index * 0.1}
            />
          ))}
        </div>
      </main>

    </div>
  );
}
