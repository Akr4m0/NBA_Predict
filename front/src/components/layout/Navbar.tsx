import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart3,
  Brain,
  CheckCircle,
  FileUp,
  Home,
  Info,
  LayoutDashboard,
  Menu,
  TrendingUp,
  Trophy,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Logo } from "@/components/shared/Logo";
import { cn } from "@/lib/utils";

const navLinks = [
  { href: "/", label: "Home", icon: Home },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/import", label: "Import", icon: FileUp },
  { href: "/train", label: "Train", icon: Brain },
  { href: "/predictions", label: "Predictions", icon: TrendingUp },
  { href: "/leaderboard", label: "Leaderboard", icon: Trophy },
  { href: "/analysis", label: "Analysis", icon: BarChart3 },
  { href: "/verify", label: "Verify", icon: CheckCircle },
  { href: "/about", label: "About", icon: Info },
];

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const location = useLocation();

  return (
    <nav className="fixed left-0 right-0 top-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-md">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link
            to="/"
            className="group flex items-center gap-2.5 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2"
          >
            <Logo size={36} className="transition-transform duration-300 group-hover:scale-105" />
            <span className="text-lg font-bold tracking-wide">
              NBA <span className="gradient-text">Predict</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden items-center gap-1 lg:flex">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const isActive = location.pathname === link.href;
              return (
                <Link
                  key={link.href}
                  to={link.href}
                  className={cn(
                    "relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2",
                    isActive
                      ? "bg-primary/10 text-primary"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {link.label}
                  {isActive && (
                    <motion.div
                      layoutId="navbar-indicator"
                      className="absolute bottom-0 left-0 h-0.5 w-full rounded-full bg-orange-500"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </Link>
              );
            })}
          </div>

          {/* Mobile Menu Button */}
          <Button
            variant="ghost"
            size="icon"
            className="lg:hidden focus-visible:ring-orange-500"
            onClick={() => setIsOpen(!isOpen)}
            aria-label={isOpen ? "Close menu" : "Open menu"}
          >
            {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>

      {/* Mobile Navigation */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden border-t border-border/50 bg-background lg:hidden"
          >
            <div className="container mx-auto px-4 py-4">
              <div className="grid gap-2">
                {navLinks.map((link) => {
                  const Icon = link.icon;
                  const isActive = location.pathname === link.href;
                  return (
                    <Link
                      key={link.href}
                      to={link.href}
                      onClick={() => setIsOpen(false)}
                      className={cn(
                        "flex items-center gap-3 rounded-lg px-4 py-3 text-sm font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-orange-500 focus-visible:ring-offset-2",
                        isActive
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      )}
                    >
                      <Icon className="h-5 w-5" />
                      {link.label}
                    </Link>
                  );
                })}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}
