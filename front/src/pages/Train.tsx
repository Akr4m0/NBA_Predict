import { useState } from "react";
import { motion } from "framer-motion";
import {
  Brain,
  ChevronDown,
  Database,
  Play,
  Settings,
  TreeDeciduous,
  Trees,
  Zap,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { ModelCard } from "@/components/shared/ModelCard";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";

const models = [
  {
    id: "decision-tree",
    name: "Decision Tree",
    description: "Interpretable decision paths with feature importance analysis",
    icon: TreeDeciduous,
    accuracy: 72,
  },
  {
    id: "random-forest",
    name: "Random Forest",
    description: "Ensemble of decision trees for higher accuracy",
    icon: Trees,
    accuracy: 76,
  },
  {
    id: "xgboost",
    name: "XGBoost",
    description: "State-of-the-art gradient boosting algorithm",
    icon: Zap,
    accuracy: 78,
  },
  {
    id: "neural-network",
    name: "Neural Network",
    description: "Deep learning for complex pattern recognition",
    icon: Brain,
    accuracy: 74,
  },
];

const datasets = [
  {
    id: "2024-season",
    name: "2024 Season",
    games: 1230,
    dateRange: "Oct 2023 - Jun 2024",
  },
  {
    id: "2023-season",
    name: "2023 Season",
    games: 1312,
    dateRange: "Oct 2022 - Jun 2023",
  },
  {
    id: "playoffs-2024",
    name: "2024 Playoffs",
    games: 89,
    dateRange: "Apr 2024 - Jun 2024",
  },
];

export default function Train() {
  const [selectedModels, setSelectedModels] = useState<string[]>(["xgboost"]);
  const [selectedDataset, setSelectedDataset] = useState("2024-season");
  const [temporalSplit, setTemporalSplit] = useState(true);
  const [crossValidation, setCrossValidation] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const { toast } = useToast();

  const toggleModel = (id: string) => {
    setSelectedModels((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  };

  const handleTrain = () => {
    if (selectedModels.length === 0) {
      toast({
        title: "No models selected",
        description: "Please select at least one model to train",
        variant: "destructive",
      });
      return;
    }

    setIsTraining(true);
    setProgress(0);

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsTraining(false);
          toast({
            title: "Training complete!",
            description: `Successfully trained ${selectedModels.length} model(s)`,
          });
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 500);
  };

  const currentDataset = datasets.find((d) => d.id === selectedDataset);

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
          <h1 className="text-3xl font-bold md:text-4xl">Train Models</h1>
          <p className="mt-2 text-muted-foreground">
            Select datasets and configure machine learning models for training
          </p>
        </motion.div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-8">
            {/* Dataset Selection */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="glass-card p-6 space-y-4"
            >
              <div className="flex items-center gap-2">
                <Database className="h-5 w-5 text-primary" />
                <h2 className="text-xl font-semibold">Select Dataset</h2>
              </div>

              <div className="grid gap-3 sm:grid-cols-3">
                {datasets.map((dataset) => (
                  <button
                    key={dataset.id}
                    onClick={() => setSelectedDataset(dataset.id)}
                    className={`rounded-lg border p-4 text-left transition-all duration-300 ${
                      selectedDataset === dataset.id
                        ? "border-primary bg-primary/10"
                        : "border-border/50 hover:border-primary/50 hover:bg-muted/30"
                    }`}
                  >
                    <p className="font-semibold">{dataset.name}</p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {dataset.games.toLocaleString()} games
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {dataset.dateRange}
                    </p>
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Model Selection */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="space-y-4"
            >
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-primary" />
                <h2 className="text-xl font-semibold">Select Models</h2>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                {models.map((model, index) => (
                  <ModelCard
                    key={model.id}
                    name={model.name}
                    description={model.description}
                    icon={model.icon}
                    accuracy={model.accuracy}
                    isSelected={selectedModels.includes(model.id)}
                    onSelect={() => toggleModel(model.id)}
                    delay={0.3 + index * 0.1}
                  />
                ))}
              </div>
            </motion.div>
          </div>

          {/* Sidebar */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="space-y-6"
          >
            {/* Training Options */}
            <div className="glass-card p-6 space-y-6">
              <div className="flex items-center gap-2">
                <Settings className="h-5 w-5 text-primary" />
                <h3 className="font-semibold">Training Options</h3>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Temporal Split</p>
                    <p className="text-sm text-muted-foreground">
                      Chronological train/test split
                    </p>
                  </div>
                  <Switch checked={temporalSplit} onCheckedChange={setTemporalSplit} />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-medium">Cross Validation</p>
                    <p className="text-sm text-muted-foreground">
                      K-fold validation for robustness
                    </p>
                  </div>
                  <Switch
                    checked={crossValidation}
                    onCheckedChange={setCrossValidation}
                  />
                </div>
              </div>

              <details className="group">
                <summary className="flex cursor-pointer items-center justify-between text-sm font-medium text-muted-foreground hover:text-foreground transition-colors">
                  Advanced Options
                  <ChevronDown className="h-4 w-4 transition-transform group-open:rotate-180" />
                </summary>
                <div className="mt-4 space-y-3 text-sm text-muted-foreground">
                  <p>• Test split ratio: 20%</p>
                  <p>• Random seed: 42</p>
                  <p>• Feature scaling: StandardScaler</p>
                </div>
              </details>
            </div>

            {/* Summary */}
            <div className="glass-card p-6 space-y-4">
              <h3 className="font-semibold">Training Summary</h3>

              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Dataset</span>
                  <span className="font-medium">{currentDataset?.name}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Games</span>
                  <span className="font-medium stats-text">
                    {currentDataset?.games.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Models</span>
                  <span className="font-medium">{selectedModels.length} selected</span>
                </div>
              </div>

              {isTraining && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="font-medium stats-text">{Math.round(progress)}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}

              <Button
                onClick={handleTrain}
                disabled={isTraining}
                variant="hero"
                size="lg"
                className="w-full"
              >
                {isTraining ? (
                  <>Training...</>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    Start Training
                  </>
                )}
              </Button>
            </div>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  );
}
