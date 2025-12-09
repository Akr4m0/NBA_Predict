"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { Upload, BrainCircuit, BarChart3, LayoutDashboard, CheckCircle2, Settings } from "lucide-react";

export default function Dashboard() {
    return (
        <div className="min-h-screen bg-[#0a0f1c] pt-20">
            <div className="container mx-auto px-4 py-12">
                <motion.div
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <div className="flex justify-between items-center mb-8">
                        <h1 className="text-4xl font-bold text-white tracking-wide">
                            NBA Prediction Dashboard
                        </h1>
                        <Link href="/">
                            <Button variant="outline" className="text-white border-white/20 hover:bg-white/10 hover:border-white/40">
                                ← Back to Home
                            </Button>
                        </Link>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {[
                            { title: "Import Data", icon: Upload, link: "/import" },
                            { title: "Train Models", icon: BrainCircuit, link: "/train" },
                            { title: "Analysis", icon: BarChart3, link: "/analysis" },
                            { title: "Dashboard", icon: LayoutDashboard, link: "/dashboard" },
                            { title: "Verification", icon: CheckCircle2, link: "/verify" },
                            { title: "Settings", icon: Settings, link: "#" },
                        ].map((item, i) => {
                            const Icon = item.icon;
                            return (
                                <Link key={i} href={item.link}>
                                    <motion.div
                                        initial={{ opacity: 0, y: 20 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        transition={{ delay: i * 0.1 }}
                                        className="backdrop-blur-md bg-white/5 hover:bg-white/10 rounded-xl p-8 border border-white/10
                                        hover:border-orange-500/50 transition-all duration-300 hover:-translate-y-2
                                        hover:shadow-xl hover:shadow-orange-500/20 cursor-pointer group"
                                    >
                                        <Icon className="w-14 h-14 mb-4 text-orange-500 group-hover:scale-110 transition-transform" strokeWidth={1.5} />
                                        <h3 className="text-2xl font-semibold text-white">
                                            {item.title}
                                        </h3>
                                    </motion.div>
                                </Link>
                            );
                        })}
                    </div>

                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.8 }}
                        className="mt-12 backdrop-blur-md bg-white/5 rounded-xl p-8 border border-white/10"
                    >
                        <h2 className="text-2xl font-bold text-white mb-6 tracking-wide">
                            Quick Stats
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div className="text-center bg-white/5 rounded-lg p-6 border border-white/10">
                                <div className="text-5xl font-bold text-orange-500 mb-2">
                                    0
                                </div>
                                <div className="text-gray-400 font-medium">
                                    Games Analyzed
                                </div>
                            </div>
                            <div className="text-center bg-white/5 rounded-lg p-6 border border-white/10">
                                <div className="text-5xl font-bold text-green-500 mb-2">
                                    0%
                                </div>
                                <div className="text-gray-400 font-medium">Accuracy</div>
                            </div>
                            <div className="text-center bg-white/5 rounded-lg p-6 border border-white/10">
                                <div className="text-5xl font-bold text-blue-500 mb-2">
                                    0
                                </div>
                                <div className="text-gray-400 font-medium">
                                    Models Trained
                                </div>
                            </div>
                        </div>
                    </motion.div>
                </motion.div>
            </div>
        </div>
    );
}
