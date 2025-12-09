"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { CheckCircle2, XCircle, TrendingUp } from "lucide-react";

export default function VerifyPage() {
    return (
        <div className="min-h-screen bg-[#0a0f1c] pt-24 pb-12">
            <div className="container mx-auto px-4">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="max-w-4xl mx-auto"
                >
                    <div className="text-center mb-12">
                        <div className="inline-flex items-center justify-center w-20 h-20 bg-orange-500/20 rounded-full mb-6">
                            <CheckCircle2 className="w-10 h-10 text-orange-500" strokeWidth={2} />
                        </div>
                        <h1 className="text-5xl font-bold text-white mb-4 tracking-wide">
                            Real Data Verification
                        </h1>
                        <p className="text-xl text-gray-400">
                            Compare predictions with actual game results
                        </p>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10 mb-8">
                        <div className="space-y-6">
                            <div>
                                <label className="block text-white font-medium mb-2">
                                    Predictions Dataset
                                </label>
                                <select className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-orange-500 transition-colors">
                                    <option>
                                        Select dataset with predictions...
                                    </option>
                                    <option>2023 Season Predictions</option>
                                    <option>2022 Season Predictions</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-white font-medium mb-2">
                                    Actual Results Dataset
                                </label>
                                <select className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-orange-500 transition-colors">
                                    <option>
                                        Select dataset with real results...
                                    </option>
                                    <option>2023 Actual Results</option>
                                    <option>2022 Actual Results</option>
                                </select>
                            </div>

                            <Button className="w-full bg-orange-500 hover:bg-orange-400 text-white py-6 text-lg font-semibold flex items-center justify-center gap-2">
                                <CheckCircle2 className="w-5 h-5" strokeWidth={2} />
                                Verify Predictions
                            </Button>
                        </div>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10">
                        <h3 className="text-xl font-semibold text-white mb-6">
                            Verification Results
                        </h3>

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                            {[
                                {
                                    label: "Correct Predictions",
                                    value: "--",
                                    icon: CheckCircle2,
                                    color: "green",
                                },
                                {
                                    label: "Total Games",
                                    value: "--",
                                    icon: TrendingUp,
                                    color: "blue",
                                },
                                {
                                    label: "Accuracy",
                                    value: "--",
                                    icon: TrendingUp,
                                    color: "orange",
                                },
                            ].map((metric, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, scale: 0.9 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="bg-white/5 rounded-xl p-6 border border-white/10 text-center"
                                >
                                    <metric.icon className="w-12 h-12 text-orange-500 mx-auto mb-4" strokeWidth={2} />
                                    <div className="text-4xl font-bold text-white mb-2">
                                        {metric.value}
                                    </div>
                                    <div className="text-gray-400">
                                        {metric.label}
                                    </div>
                                </motion.div>
                            ))}
                        </div>

                        <div className="text-center py-8">
                            <CheckCircle2 className="w-16 h-16 text-orange-500/30 mx-auto mb-4" strokeWidth={1.5} />
                            <p className="text-gray-400">
                                Select datasets and click "Verify Predictions" to
                                compare results
                            </p>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
