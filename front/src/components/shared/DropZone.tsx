import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileSpreadsheet, X, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface DropZoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
  maxSize?: number; // in MB
}

export function DropZone({
  onFileSelect,
  accept = ".csv,.xlsx,.xls",
  maxSize = 50,
}: DropZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  }, []);

  const validateFile = (file: File): boolean => {
    const validExtensions = accept.split(",").map((ext) => ext.trim());
    const fileExtension = `.${file.name.split(".").pop()?.toLowerCase()}`;

    if (!validExtensions.includes(fileExtension)) {
      setError(`Invalid file type. Accepted: ${accept}`);
      return false;
    }

    if (file.size > maxSize * 1024 * 1024) {
      setError(`File too large. Maximum size: ${maxSize}MB`);
      return false;
    }

    setError(null);
    return true;
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file && validateFile(file)) {
        setSelectedFile(file);
        onFileSelect(file);
      }
    },
    [onFileSelect, accept, maxSize]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && validateFile(file)) {
      setSelectedFile(file);
      onFileSelect(file);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setError(null);
  };

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      className={cn(
        "relative rounded-xl border-2 border-dashed p-8 transition-all duration-300",
        isDragging
          ? "border-primary bg-primary/10"
          : selectedFile
          ? "border-success bg-success/5"
          : "border-border hover:border-primary/50 hover:bg-muted/30"
      )}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleChange}
        className="absolute inset-0 cursor-pointer opacity-0"
      />

      <AnimatePresence mode="wait">
        {selectedFile ? (
          <motion.div
            key="selected"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="flex flex-col items-center gap-4"
          >
            <div className="flex h-16 w-16 items-center justify-center rounded-xl bg-success/10">
              <CheckCircle className="h-8 w-8 text-success" />
            </div>
            <div className="text-center">
              <p className="font-semibold">{selectedFile.name}</p>
              <p className="text-sm text-muted-foreground">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.preventDefault();
                clearFile();
              }}
              className="gap-2"
            >
              <X className="h-4 w-4" />
              Remove
            </Button>
          </motion.div>
        ) : (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-4"
          >
            <div
              className={cn(
                "flex h-16 w-16 items-center justify-center rounded-xl transition-all duration-300",
                isDragging ? "bg-primary/20" : "bg-muted"
              )}
            >
              {isDragging ? (
                <Upload className="h-8 w-8 text-primary" />
              ) : (
                <FileSpreadsheet className="h-8 w-8 text-muted-foreground" />
              )}
            </div>
            <div className="text-center">
              <p className="font-semibold">
                {isDragging ? "Drop your file here" : "Drag & drop your file here"}
              </p>
              <p className="text-sm text-muted-foreground">
                or click to browse
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {accept.split(",").map((ext) => (
                <span
                  key={ext}
                  className="rounded-full bg-secondary/20 px-3 py-1 text-xs font-medium text-secondary"
                >
                  {ext.trim().toUpperCase()}
                </span>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {error && (
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mt-4 text-center text-sm text-destructive"
        >
          {error}
        </motion.p>
      )}
    </div>
  );
}
