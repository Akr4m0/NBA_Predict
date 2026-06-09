import { Link } from "react-router-dom";
import { Github, Twitter } from "lucide-react";
import { Logo } from "@/components/shared/Logo";

export function Footer() {
  return (
    <footer className="border-t border-border/50 bg-card/30">
      <div className="container mx-auto px-4 py-12">
        <div className="grid gap-8 md:grid-cols-4">
          {/* Brand */}
          <div className="space-y-4">
            <Link to="/" className="flex items-center gap-2.5">
              <Logo size={36} />
              <span className="text-lg font-bold tracking-wide">
                NBA <span className="gradient-text">Predict</span>
              </span>
            </Link>
            <p className="text-sm text-muted-foreground">
              Machine learning powered NBA game predictions for data-driven insights.
            </p>
          </div>

          {/* Navigation */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold uppercase tracking-wider">Navigation</h4>
            <div className="flex flex-col gap-2">
              <Link to="/dashboard" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Dashboard
              </Link>
              <Link to="/predictions" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Predictions
              </Link>
              <Link to="/analysis" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Analysis
              </Link>
            </div>
          </div>

          {/* Tools */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold uppercase tracking-wider">Tools</h4>
            <div className="flex flex-col gap-2">
              <Link to="/import" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Import Data
              </Link>
              <Link to="/train" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Train Models
              </Link>
              <Link to="/verify" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
                Verify Results
              </Link>
            </div>
          </div>

          {/* Social */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold uppercase tracking-wider">Connect</h4>
            <div className="flex gap-4">
              <a
                href="#"
                className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted text-muted-foreground transition-all duration-300 hover:bg-primary hover:text-primary-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="#"
                className="flex h-10 w-10 items-center justify-center rounded-lg bg-muted text-muted-foreground transition-all duration-300 hover:bg-primary hover:text-primary-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2"
              >
                <Twitter className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>

        <div className="mt-12 border-t border-border/50 pt-8">
          <p className="text-center text-sm text-muted-foreground">
            © {new Date().getFullYear()} NBA Predict. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
}
