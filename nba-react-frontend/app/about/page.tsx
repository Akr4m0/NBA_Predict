"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Activity, Sparkles, BarChart3, BrainCircuit, TrendingUp, LayoutDashboard, CheckCircle2, Microscope, Wrench } from "lucide-react";

export default function About() {
    return (
        <div className="min-h-screen bg-[#0a0f1c] pt-20">
            <div className="container mx-auto px-4 py-12 max-w-4xl">
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <div className="flex justify-between items-center mb-8">
                        <h1 className="text-4xl font-bold text-white tracking-wide">
                            About NBA Prediction System
                        </h1>
                        <Link href="/">
                            <Button variant="outline" className="text-white border-white/20 hover:bg-white/10 hover:border-white/40">
                                ← Home
                            </Button>
                        </Link>
                    </div>

                    <div className="space-y-6">
                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.2 }}
                            className="backdrop-blur-md bg-white/5 rounded-xl p-8 border border-white/10"
                        >
                            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                                <Activity className="w-7 h-7 text-orange-500" strokeWidth={2} />
                                Overview
                            </h2>
                            <p className="text-gray-400 leading-relaxed">
                                A comprehensive NBA game prediction system that
                                allows you to import historical data, train
                                machine learning models, compare their
                                performance, and verify predictions against real
                                results.
                            </p>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="backdrop-blur-md bg-white/5 rounded-xl p-8 border border-white/10"
                        >
                            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                                <Sparkles className="w-7 h-7 text-orange-500" strokeWidth={2} />
                                Features
                            </h2>
                            <ul className="space-y-3 text-gray-400">
                                <li className="flex items-start gap-3">
                                    <BarChart3 className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" strokeWidth={2} />
                                    <span>
                                        <strong className="text-white">Historical Data Import:</strong>{" "}
                                        Import NBA game data from CSV/Excel
                                        files
                                    </span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <BrainCircuit className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" strokeWidth={2} />
                                    <span>
                                        <strong className="text-white">Multiple ML Models:</strong>{" "}
                                        Decision Tree and Random Forest
                                        classifiers
                                    </span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <TrendingUp className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" strokeWidth={2} />
                                    <span>
                                        <strong className="text-white">Performance Evaluation:</strong>{" "}
                                        Comprehensive metrics including
                                        accuracy, precision, recall, and
                                        F1-score
                                    </span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <LayoutDashboard className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" strokeWidth={2} />
                                    <span>
                                        <strong className="text-white">Interactive Dashboard:</strong>{" "}
                                        Web-based visualization and analysis
                                        tool
                                    </span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <CheckCircle2 className="w-5 h-5 text-orange-500 mt-0.5 flex-shrink-0" strokeWidth={2} />
                                    <span>
                                        <strong className="text-white">Real Data Verification:</strong>{" "}
                                        Compare predictions with actual game
                                        results
                                    </span>
                                </li>
                            </ul>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.4 }}
                            className="backdrop-blur-md bg-white/5 rounded-xl p-8 border border-white/10"
                        >
                            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                                <Microscope className="w-7 h-7 text-orange-500" strokeWidth={2} />
                                ML Models
                            </h2>
                            <div className="space-y-4">
                                <div>
                                    <h3 className="text-xl font-semibold text-orange-400 mb-2">
                                        Decision Tree Classifier
                                    </h3>
                                    <p className="text-gray-400">
                                        Interpretable model showing decision
                                        paths. Good for understanding feature
                                        importance and making transparent
                                        predictions.
                                    </p>
                                </div>
                                <div>
                                    <h3 className="text-xl font-semibold text-orange-400 mb-2">
                                        Random Forest Classifier
                                    </h3>
                                    <p className="text-gray-400">
                                        Ensemble method combining multiple
                                        trees. Generally higher accuracy and
                                        robust to overfitting.
                                    </p>
                                </div>
                            </div>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 }}
                            className="backdrop-blur-md bg-white/5 rounded-xl p-8 border border-white/10"
                        >
                            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                                <Wrench className="w-7 h-7 text-orange-500" strokeWidth={2} />
                                Technology Stack
                            </h2>
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                {[
                                    "Python",
                                    "Next.js",
                                    "TypeScript",
                                    "Tailwind CSS",
                                    "Scikit-learn",
                                    "Framer Motion",
                                    "SQLite",
                                    "Pandas",
                                    "shadcn/ui",
                                ].map((tech, i) => (
                                    <div
                                        key={i}
                                        className="bg-white/5 rounded-lg p-3 text-center text-gray-300 border border-white/10 hover:border-orange-500/50 hover:bg-white/10 transition-all"
                                    >
                                        {tech}
                                    </div>
                                ))}
                            </div>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.6 }}
                            className="text-center pt-8"
                        >
                            <Link href="/dashboard">
                                <Button
                                    size="lg"
                                    className="bg-orange-500 hover:bg-orange-400 text-white px-8 py-6 text-lg"
                                >
                                    Get Started →
                                </Button>
                            </Link>
                        </motion.div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
