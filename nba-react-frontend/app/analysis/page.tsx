"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { BarChart3, TrendingUp, Award, PieChart } from "lucide-react";

export default function AnalysisPage() {
    return (
        <div className="min-h-screen bg-[#0a0f1c] pt-24 pb-12">
            <div className="container mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="max-w-6xl mx-auto"
                >
                    <div className="text-center mb-12">
                        <div className="inline-flex items-center justify-center w-20 h-20 bg-orange-500/20 rounded-full mb-6">
                            <BarChart3 className="w-10 h-10 text-orange-500" strokeWidth={2} />
                        </div>
                        <h1 className="text-5xl font-bold text-white mb-4 tracking-wide">
                            Performance Analysis
                        </h1>
                        <p className="text-xl text-gray-400">
                            Compare model performance and analyze results
                        </p>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10 mb-8">
                        <div className="flex gap-4 mb-6">
                            <select className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-orange-500 transition-colors">
                                <option>All datasets</option>
                                <option>2023 NBA Season</option>
                                <option>2022 NBA Season</option>
                            </select>
                            <Button className="bg-orange-500 hover:bg-orange-400 font-semibold">
                                Compare Models
                            </Button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                            {[
                                {
                                    label: "Accuracy",
                                    value: "--",
                                    icon: TrendingUp,
                                    color: "blue",
                                },
                                {
                                    label: "Precision",
                                    value: "--",
                                    icon: Award,
                                    color: "green",
                                },
                                {
                                    label: "Recall",
                                    value: "--",
                                    icon: PieChart,
                                    color: "purple",
                                },
                                {
                                    label: "F1-Score",
                                    value: "--",
                                    icon: BarChart3,
                                    color: "orange",
                                },
                            ].map((metric, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="bg-white/5 rounded-xl p-6 border border-white/10"
                                >
                                    <div className="flex items-center justify-between mb-4">
                                        <metric.icon className="w-8 h-8 text-orange-500" strokeWidth={2} />
                                    </div>
                                    <div className="text-3xl font-bold text-white mb-2">
                                        {metric.value}
                                    </div>
                                    <div className="text-gray-400 text-sm">
                                        {metric.label}
                                    </div>
                                </motion.div>
                            ))}
                        </div>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10 mb-8">
                        <h3 className="text-xl font-semibold text-white mb-4">
                            Model Comparison
                        </h3>
                        <div className="text-center py-12">
                            <BarChart3 className="w-16 h-16 text-orange-500/30 mx-auto mb-4" strokeWidth={1.5} />
                            <p className="text-gray-400">
                                Click "Compare Models" to see detailed analysis
                            </p>
                        </div>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10">
                        <h3 className="text-xl font-semibold text-white mb-4">
                            Generate Report
                        </h3>
                        <div className="flex gap-4">
                            <select className="flex-1 px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-orange-500 transition-colors">
                                <option>Choose a dataset...</option>
                                <option>2023 NBA Season</option>
                                <option>2022 NBA Season</option>
                            </select>
                            <Button className="bg-orange-500 hover:bg-orange-400 font-semibold flex items-center gap-2">
                                <PieChart className="w-4 h-4" strokeWidth={2} />
                                Generate Report
                            </Button>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
