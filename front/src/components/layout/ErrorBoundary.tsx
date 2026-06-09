import React from "react";
import { AlertTriangle } from "lucide-react";

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  State
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error("ErrorBoundary caught:", error, info);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-background p-8">
          <div className="w-full max-w-md space-y-6 rounded-2xl border border-white/10 bg-white/5 p-10 text-center backdrop-blur-md">
            <div className="flex justify-center">
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-destructive/10">
                <AlertTriangle className="h-8 w-8 text-destructive" />
              </div>
            </div>

            <div className="space-y-2">
              <h2 className="text-xl font-semibold">Something went wrong</h2>
              {this.state.error && (
                <p className="rounded-lg bg-muted/30 px-4 py-3 font-mono text-sm text-muted-foreground">
                  {this.state.error.message}
                </p>
              )}
            </div>

            <button
              onClick={() => window.location.reload()}
              className="inline-flex items-center justify-center rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2"
            >
              Reload page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
