"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Upload, FileSpreadsheet, Database } from "lucide-react";

export default function ImportPage() {
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
                            <Upload className="w-10 h-10 text-orange-500" strokeWidth={2} />
                        </div>
                        <h1 className="text-5xl font-bold text-white mb-4 tracking-wide">
                            Import Historical Data
                        </h1>
                        <p className="text-xl text-gray-400">
                            Upload NBA game data from CSV or Excel files to begin training your models
                        </p>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10 mb-8">
                        <div className="border-2 border-dashed border-orange-400/50 rounded-xl p-12 text-center hover:border-orange-400 transition-colors cursor-pointer">
                            <FileSpreadsheet className="w-16 h-16 text-orange-500 mx-auto mb-4" strokeWidth={1.5} />
                            <h3 className="text-xl font-semibold text-white mb-2">
                                Drop your file here or click to browse
                            </h3>
                            <p className="text-gray-400 mb-4">
                                Supports CSV and Excel formats (.csv, .xlsx, .xls)
                            </p>
                            <Button className="bg-orange-500 hover:bg-orange-400">
                                Select File
                            </Button>
                        </div>

                        <div className="mt-6">
                            <label className="block text-white font-medium mb-2">
                                Dataset Description
                            </label>
                            <input
                                type="text"
                                placeholder="e.g., 2023 NBA Regular Season"
                                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-orange-500 transition-colors"
                            />
                        </div>
                    </div>

                    <div className="backdrop-blur-md bg-white/5 rounded-2xl p-8 border border-white/10">
                        <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
                            <Database className="w-6 h-6 text-orange-500" strokeWidth={2} />
                            Required Data Format
                        </h3>
                        <div className="space-y-4 text-gray-400">
                            <div>
                                <h4 className="font-semibold text-white mb-2">
                                    Required Columns:
                                </h4>
                                <ul className="space-y-1 ml-4">
                                    <li>
                                        <code className="bg-slate-800 px-2 py-1 rounded">
                                            home_team
                                        </code>{" "}
                                        - Home team name
                                    </li>
                                    <li>
                                        <code className="bg-slate-800 px-2 py-1 rounded">
                                            away_team
                                        </code>{" "}
                                        - Away team name
                                    </li>
                                    <li>
                                        <code className="bg-slate-800 px-2 py-1 rounded">
                                            game_date
                                        </code>{" "}
                                        - Date of the game
                                    </li>
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold text-white mb-2">
                                    Optional but Recommended:
                                </h4>
                                <ul className="space-y-1 ml-4">
                                    <li>
                                        <code className="bg-slate-800 px-2 py-1 rounded">
                                            home_score
                                        </code>
                                        ,{" "}
                                        <code className="bg-slate-800 px-2 py-1 rounded">
                                            away_score
                                        </code>{" "}
                                        - Game scores
                                    </li>
                                    <li>
                                        <code className="bg-slate-800 px-2 py-1 rounded">
                                            season
                                        </code>{" "}
                                        - Season identifier
                                    </li>
                                    <li>Team statistics (FG%, rebounds, assists, etc.)</li>
                                </ul>
                            </div>
                        </div>

                        <div className="mt-6">
                            <Button className="w-full bg-orange-500 hover:bg-orange-400 text-white py-6 text-lg font-semibold">
                                Import Dataset
                            </Button>
                        </div>
                    </div>
                </motion.div>
            </div>
        </div>
    );
}
