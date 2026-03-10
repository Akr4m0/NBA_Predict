import { motion } from "framer-motion";
import {
  Brain,
  ChevronDown,
  Code,
  Database,
  FileUp,
  TrendingUp,
  Zap,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const steps = [
  {
    icon: FileUp,
    title: "Import Data",
    description:
      "Upload historical NBA game data in CSV or Excel format. The system automatically validates and cleans your data.",
  },
  {
    icon: Brain,
    title: "Train Models",
    description:
      "Choose from multiple ML algorithms and configure training parameters. The system learns patterns from your data.",
  },
  {
    icon: TrendingUp,
    title: "Generate Predictions",
    description:
      "Apply trained models to upcoming games and receive predictions with confidence scores.",
  },
  {
    icon: Zap,
    title: "Verify & Improve",
    description:
      "Compare predictions with actual results to track accuracy and continuously improve model performance.",
  },
];

const techStack = [
  { name: "Python", category: "Backend" },
  { name: "XGBoost", category: "ML" },
  { name: "Scikit-learn", category: "ML" },
  { name: "Pandas", category: "Data" },
  { name: "SQLite", category: "Database" },
  { name: "React", category: "Frontend" },
  { name: "TypeScript", category: "Frontend" },
  { name: "Tailwind CSS", category: "Styling" },
  { name: "Recharts", category: "Charts" },
  { name: "Framer Motion", category: "Animation" },
];

const faqs = [
  {
    question: "What data format is required for import?",
    answer:
      "The system accepts CSV and Excel files (XLSX, XLS) with columns for Date, Home Team, Away Team, Home Score, and Away Score. Additional statistical columns are optional but improve prediction accuracy.",
  },
  {
    question: "How accurate are the predictions?",
    answer:
      "Our best-performing model (XGBoost) achieves approximately 78% accuracy on historical data. Actual performance may vary based on the quality and recency of training data.",
  },
  {
    question: "Which machine learning models are available?",
    answer:
      "Currently, the system supports Decision Trees, Random Forest, XGBoost, and Neural Networks. Each model has different strengths - XGBoost typically offers the best accuracy while Decision Trees provide the most interpretable results.",
  },
  {
    question: "Can I use my own custom features?",
    answer:
      "Yes! The system automatically extracts features from your data, but you can also include custom columns in your import file. The feature engineering pipeline will incorporate them into the training process.",
  },
  {
    question: "How often should I retrain models?",
    answer:
      "We recommend retraining monthly or when you add significant new data. Regular retraining helps capture recent trends and improves prediction accuracy for upcoming games.",
  },
];

export default function About() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      <main className="container mx-auto px-4 pt-24 pb-12">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="mx-auto max-w-3xl text-center mb-16"
        >
          <h1 className="text-3xl font-bold md:text-4xl">About NBA Predict</h1>
          <p className="mt-4 text-lg text-muted-foreground">
            A machine learning powered platform for predicting NBA game outcomes
            with professional-grade accuracy and comprehensive analytics.
          </p>
        </motion.div>

        {/* How It Works */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="mb-20"
        >
          <h2 className="mb-8 text-center text-2xl font-bold">How It Works</h2>
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {steps.map((step, index) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 + index * 0.1 }}
                className="glass-card p-6 text-center"
              >
                <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10">
                  <step.icon className="h-7 w-7 text-primary" />
                </div>
                <div className="mb-2 flex items-center justify-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                    {index + 1}
                  </span>
                  <h3 className="font-semibold">{step.title}</h3>
                </div>
                <p className="text-sm text-muted-foreground">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.section>

        {/* Tech Stack */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mb-20"
        >
          <h2 className="mb-8 text-center text-2xl font-bold">Technology Stack</h2>
          <div className="glass-card p-8">
            <div className="flex flex-wrap justify-center gap-3">
              {techStack.map((tech, index) => (
                <motion.div
                  key={tech.name}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: 0.7 + index * 0.05 }}
                  className="group relative"
                >
                  <div className="rounded-lg border border-border/50 bg-muted/30 px-4 py-2 text-sm font-medium transition-all duration-300 hover:border-primary/50 hover:bg-primary/10">
                    {tech.name}
                  </div>
                  <span className="absolute -top-2 left-1/2 -translate-x-1/2 opacity-0 transition-all duration-300 group-hover:-top-8 group-hover:opacity-100">
                    <span className="rounded bg-card px-2 py-1 text-xs text-muted-foreground shadow-lg">
                      {tech.category}
                    </span>
                  </span>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>

        {/* FAQ */}
        <motion.section
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.8 }}
          className="mx-auto max-w-3xl"
        >
          <h2 className="mb-8 text-center text-2xl font-bold">
            Frequently Asked Questions
          </h2>
          <Accordion type="single" collapsible className="space-y-4">
            {faqs.map((faq, index) => (
              <motion.div
                key={faq.question}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.9 + index * 0.1 }}
              >
                <AccordionItem
                  value={`item-${index}`}
                  className="glass-card overflow-hidden border-none px-6"
                >
                  <AccordionTrigger className="text-left hover:no-underline py-6">
                    {faq.question}
                  </AccordionTrigger>
                  <AccordionContent className="pb-6 text-muted-foreground">
                    {faq.answer}
                  </AccordionContent>
                </AccordionItem>
              </motion.div>
            ))}
          </Accordion>
        </motion.section>
      </main>

      <Footer />
    </div>
  );
}
