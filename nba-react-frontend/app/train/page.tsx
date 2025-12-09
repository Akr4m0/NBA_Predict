"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { BrainCircuit, Zap, Target } from "lucide-react";

export default function TrainPage() {
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
                            <BrainCircuit className="w-10 h-10 text-orange-500" strokeWidth={2} />
                        </div>
                        <h1 className="text-5xl font-bold text-white mb-4 tracking-wide">
                            Train Prediction Models
                        </h1>
                        <p className="text-xl text-gray-400">
                            Select your dataset and train machine learning models
                        </p>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10 mb-8">
                        <div className="mb-6">
                            <label className="block text-white font-medium mb-2">
                                Select Dataset
                            </label>
                            <select className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-orange-500 transition-colors">
                                <option>Choose a dataset...</option>
                                <option>2023 NBA Season (1,230 games)</option>
                                <option>2022 NBA Season (1,200 games)</option>
                            </select>
                        </div>

                        <div>
                            <h3 className="text-xl font-semibold text-white mb-4">
                                Choose Models to Train
                            </h3>
                            <div className="space-y-4">
                                <label className="flex items-start gap-4 p-4 bg-white/5 rounded-lg border-2 border-white/10 hover:border-orange-500/50 cursor-pointer transition-all">
                                    <input
                                        type="checkbox"
                                        defaultChecked
                                        className="mt-1 w-5 h-5 accent-orange-600"
                                    />
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Zap className="w-5 h-5 text-orange-500" strokeWidth={2} />
                                            <h4 className="font-semibold text-white">
                                                Decision Tree Classifier
                                            </h4>
                                        </div>
                                        <p className="text-gray-400 text-sm">
                                            Interpretable model showing decision paths.
                                            Good for understanding feature importance.
                                        </p>
                                        <div className="mt-2">
                                            <span className="inline-block px-3 py-1 bg-orange-500/20 text-orange-300 text-xs rounded-full">
                                                Fast Training
                                            </span>
                                        </div>
                                    </div>
                                </label>

                                <label className="flex items-start gap-4 p-4 bg-white/5 rounded-lg border-2 border-white/10 hover:border-orange-500/50 cursor-pointer transition-all">
                                    <input
                                        type="checkbox"
                                        defaultChecked
                                        className="mt-1 w-5 h-5 accent-orange-600"
                                    />
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-2">
                                            <Target className="w-5 h-5 text-green-500" strokeWidth={2} />
                                            <h4 className="font-semibold text-white">
                                                Random Forest Classifier
                                            </h4>
                                        </div>
                                        <p className="text-gray-400 text-sm">
                                            Ensemble method combining multiple trees.
                                            Generally higher accuracy and robust to
                                            overfitting.
                                        </p>
                                        <div className="mt-2">
                                            <span className="inline-block px-3 py-1 bg-green-500/20 text-green-300 text-xs rounded-full">
                                                High Accuracy
                                            </span>
                                        </div>
                                    </div>
                                </label>
                            </div>
                        </div>

                        <div className="mt-8">
                            <Button className="w-full bg-orange-500 hover:bg-orange-400 text-white py-6 text-lg font-semibold flex items-center justify-center gap-2">
                                <BrainCircuit className="w-5 h-5" strokeWidth={2} />
                                Start Training
                            </Button>
                        </div>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10">
                        <h3 className="text-xl font-semibold text-white mb-4">
                            Trained Models
                        </h3>
                        <div className="text-center py-12">
                            <BrainCircuit className="w-16 h-16 text-orange-500/30 mx-auto mb-4" strokeWidth={1.5} />
                            <p className="text-gray-400">
                                No models trained yet. Import data and start training!
                            </p>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
