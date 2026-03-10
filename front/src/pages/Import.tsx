import { useState } from "react";
import { motion } from "framer-motion";
import {
  Calendar,
  FileSpreadsheet,
  Info,
  Trash2,
  Users,
} from "lucide-react";
import { Navbar } from "@/components/layout/Navbar";
import { Footer } from "@/components/layout/Footer";
import { DropZone } from "@/components/shared/DropZone";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";

const importHistory = [
  {
    id: 1,
    filename: "nba_games_2024.csv",
    games: 1230,
    teams: 30,
    dateRange: "Oct 2023 - Jun 2024",
    importedAt: "2024-01-15",
  },
  {
    id: 2,
    filename: "nba_games_2023.xlsx",
    games: 1312,
    teams: 30,
    dateRange: "Oct 2022 - Jun 2023",
    importedAt: "2024-01-10",
  },
  {
    id: 3,
    filename: "playoffs_2024.csv",
    games: 89,
    teams: 16,
    dateRange: "Apr 2024 - Jun 2024",
    importedAt: "2024-01-08",
  },
];

export default function Import() {
  const [description, setDescription] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const { toast } = useToast();

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    toast({
      title: "File selected",
      description: `${file.name} is ready for import`,
    });
  };

  const handleImport = () => {
    if (!selectedFile) return;

    toast({
      title: "Import started",
      description: "Processing your data file...",
    });

    // Simulate import
    setTimeout(() => {
      toast({
        title: "Import complete!",
        description: `Successfully imported ${selectedFile.name}`,
      });
      setSelectedFile(null);
      setDescription("");
    }, 2000);
  };

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
          <h1 className="text-3xl font-bold md:text-4xl">Import Data</h1>
          <p className="mt-2 text-muted-foreground">
            Upload historical NBA game data for training and predictions
          </p>
        </motion.div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Upload Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="lg:col-span-2 space-y-6"
          >
            <div className="glass-card p-6 space-y-6">
              <h2 className="text-xl font-semibold">Upload File</h2>

              <DropZone onFileSelect={handleFileSelect} />

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Description (optional)
                </label>
                <Input
                  placeholder="e.g., Regular season 2024 data"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="bg-muted/30 border-border/50"
                />
              </div>

              <Button
                onClick={handleImport}
                disabled={!selectedFile}
                variant="hero"
                size="lg"
                className="w-full"
              >
                Import Data
              </Button>
            </div>
          </motion.div>

          {/* Requirements Panel */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-6"
          >
            <div className="glass-card p-6 space-y-4">
              <div className="flex items-center gap-2">
                <Info className="h-5 w-5 text-secondary" />
                <h3 className="font-semibold">File Requirements</h3>
              </div>

              <div className="space-y-3 text-sm">
                <div className="flex items-start gap-3">
                  <div className="mt-1 h-1.5 w-1.5 rounded-full bg-primary" />
                  <p className="text-muted-foreground">
                    <span className="text-foreground">Formats:</span> CSV, XLSX,
                    XLS
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-1 h-1.5 w-1.5 rounded-full bg-primary" />
                  <p className="text-muted-foreground">
                    <span className="text-foreground">Required columns:</span>{" "}
                    Date, Home Team, Away Team, Home Score, Away Score
                  </p>
                </div>
                <div className="flex items-start gap-3">
                  <div className="mt-1 h-1.5 w-1.5 rounded-full bg-primary" />
                  <p className="text-muted-foreground">
                    <span className="text-foreground">Max size:</span> 50MB
                  </p>
                </div>
              </div>

              <Button variant="outline" size="sm" className="w-full">
                Download Template
              </Button>
            </div>
          </motion.div>
        </div>

        {/* Import History */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-12"
        >
          <h2 className="mb-6 text-xl font-semibold">Import History</h2>

          <div className="glass-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/50 bg-muted/30">
                    <th className="px-6 py-4 text-left text-sm font-semibold">
                      Filename
                    </th>
                    <th className="px-6 py-4 text-left text-sm font-semibold">
                      Games
                    </th>
                    <th className="px-6 py-4 text-left text-sm font-semibold">
                      Teams
                    </th>
                    <th className="px-6 py-4 text-left text-sm font-semibold">
                      Date Range
                    </th>
                    <th className="px-6 py-4 text-left text-sm font-semibold">
                      Imported
                    </th>
                    <th className="px-6 py-4 text-right text-sm font-semibold">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {importHistory.map((item, index) => (
                    <motion.tr
                      key={item.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: 0.4 + index * 0.1 }}
                      className="hover:bg-muted/30 transition-colors"
                    >
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-3">
                          <FileSpreadsheet className="h-5 w-5 text-muted-foreground" />
                          <span className="font-medium">{item.filename}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <span className="stats-text">{item.games.toLocaleString()}</span>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <Users className="h-4 w-4 text-muted-foreground" />
                          <span>{item.teams}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          <Calendar className="h-4 w-4 text-muted-foreground" />
                          <span className="text-muted-foreground">
                            {item.dateRange}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-muted-foreground">
                        {item.importedAt}
                      </td>
                      <td className="px-6 py-4 text-right">
                        <Button variant="ghost" size="icon">
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.div>
      </main>

      <Footer />
    </div>
  );
}
