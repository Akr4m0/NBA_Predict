import { motion } from "framer-motion";

interface Stat {
  label: string;
  value: string;
}

interface StatsTickerProps {
  stats: Stat[];
}

export function StatsTicker({ stats }: StatsTickerProps) {
  return (
    <div className="flex flex-wrap justify-center gap-8 md:gap-16">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: index * 0.1 + 0.5 }}
          className="text-center"
        >
          <p className="text-3xl font-bold gradient-text stats-text md:text-4xl">
            {stat.value}
          </p>
          <p className="mt-1 text-sm text-muted-foreground uppercase tracking-wider">
            {stat.label}
          </p>
        </motion.div>
      ))}
    </div>
  );
}
