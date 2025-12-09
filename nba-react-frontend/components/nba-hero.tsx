"use client";

import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { BarChart3, BrainCircuit, TrendingUp } from "lucide-react";

function FloatingPaths({ position }: { position: number }) {
    const paths = Array.from({ length: 36 }, (_, i) => ({
        id: i,
        d: `M-${380 - i * 5 * position} -${189 + i * 6}C-${
            380 - i * 5 * position
        } -${189 + i * 6} -${312 - i * 5 * position} ${216 - i * 6} ${
            152 - i * 5 * position
        } ${343 - i * 6}C${616 - i * 5 * position} ${470 - i * 6} ${
            684 - i * 5 * position
        } ${875 - i * 6} ${684 - i * 5 * position} ${875 - i * 6}`,
        color: `rgba(15,23,42,${0.1 + i * 0.03})`,
        width: 0.5 + i * 0.03,
    }));

    return (
        <div className="absolute inset-0 pointer-events-none">
            <svg
                className="w-full h-full text-slate-950 dark:text-white"
                viewBox="0 0 696 316"
                fill="none"
            >
                <title>Background Paths</title>
                {paths.map((path) => (
                    <motion.path
                        key={path.id}
                        d={path.d}
                        stroke="currentColor"
                        strokeWidth={path.width}
                        strokeOpacity={0.1 + path.id * 0.03}
                        initial={{ pathLength: 0.3, opacity: 0.6 }}
                        animate={{
                            pathLength: 1,
                            opacity: [0.3, 0.6, 0.3],
                            pathOffset: [0, 1, 0],
                        }}
                        transition={{
                            duration: 20 + Math.random() * 10,
                            repeat: Number.POSITIVE_INFINITY,
                            ease: "linear",
                        }}
                    />
                ))}
            </svg>
        </div>
    );
}

export function NBAHero() {
    const title = "NBA Prediction System";
    const words = title.split(" ");

    return (
        <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-[#0a0f1c]">
            <div className="absolute inset-0 opacity-30">
                <FloatingPaths position={1} />
                <FloatingPaths position={-1} />
            </div>

            {/* Basketball decoration - Static SVG */}
            <div className="absolute top-20 right-20 w-64 h-64 opacity-5">
                <svg viewBox="0 0 200 200" className="w-full h-full">
                    <defs>
                        <radialGradient id="basketballGradient" cx="35%" cy="35%">
                            <stop offset="0%" stopColor="#ff8c42" />
                            <stop offset="100%" stopColor="#d45d00" />
                        </radialGradient>
                    </defs>
                    <circle cx="100" cy="100" r="95" fill="url(#basketballGradient)" stroke="#000" strokeWidth="2"/>
                    <path d="M 100 5 Q 100 50 100 100 Q 100 150 100 195" stroke="#000" strokeWidth="3" fill="none"/>
                    <path d="M 5 100 Q 50 100 100 100 Q 150 100 195 100" stroke="#000" strokeWidth="3" fill="none"/>
                    <path d="M 30 30 Q 60 90 100 100 Q 140 110 170 170" stroke="#000" strokeWidth="3" fill="none"/>
                    <path d="M 170 30 Q 140 90 100 100 Q 60 110 30 170" stroke="#000" strokeWidth="3" fill="none"/>
                </svg>
            </div>

            <div className="relative z-10 container mx-auto px-4 md:px-6 text-center">
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 2 }}
                    className="max-w-5xl mx-auto"
                >
                    <h1 className="text-5xl sm:text-7xl md:text-8xl font-bold mb-8 tracking-wide">
                        {words.map((word, wordIndex) => (
                            <span
                                key={wordIndex}
                                className="inline-block mr-5 last:mr-0"
                            >
                                {word.split("").map((letter, letterIndex) => (
                                    <motion.span
                                        key={`${wordIndex}-${letterIndex}`}
                                        initial={{ y: 100, opacity: 0 }}
                                        animate={{ y: 0, opacity: 1 }}
                                        transition={{
                                            delay:
                                                wordIndex * 0.1 +
                                                letterIndex * 0.03,
                                            type: "spring",
                                            stiffness: 150,
                                            damping: 25,
                                        }}
                                        className="inline-block text-white mr-0.5"
                                        style={{ letterSpacing: '0.02em' }}
                                    >
                                        {letter}
                                    </motion.span>
                                ))}
                            </span>
                        ))}
                    </h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.8, duration: 0.8 }}
                        className="text-xl md:text-2xl text-gray-400 mb-12 max-w-3xl mx-auto font-light"
                    >
                        Advanced Decision Tree and Random Forest models trained
                        on historical NBA data. Predict game outcomes with
                        machine learning.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 1.2, duration: 0.8 }}
                        className="flex flex-col sm:flex-row gap-4 justify-center items-center"
                    >
                        <div
                            className="inline-block group relative bg-gradient-to-b from-orange-600/60 to-orange-700/60
                            p-px rounded-xl backdrop-blur-lg overflow-hidden shadow-lg hover:shadow-2xl hover:shadow-orange-500/30
                            transition-all duration-300"
                        >
                            <Link href="/dashboard">
                                <Button
                                    variant="ghost"
                                    size="lg"
                                    className="rounded-[0.9rem] px-8 py-6 text-lg font-semibold backdrop-blur-md
                                    bg-orange-500 hover:bg-orange-400
                                    text-white transition-all duration-300
                                    group-hover:-translate-y-0.5 border-0
                                    hover:shadow-md"
                                >
                                    <span className="transition-opacity">
                                        Launch Dashboard
                                    </span>
                                    <span
                                        className="ml-3 group-hover:translate-x-1.5
                                        transition-all duration-300"
                                    >
                                        →
                                    </span>
                                </Button>
                            </Link>
                        </div>

                        <div
                            className="inline-block group relative
                            p-px rounded-xl backdrop-blur-lg overflow-hidden
                            transition-all duration-300"
                        >
                            <Link href="/about">
                                <Button
                                    variant="ghost"
                                    size="lg"
                                    className="rounded-xl px-8 py-6 text-lg font-semibold backdrop-blur-md
                                    bg-white/5 hover:bg-white/10
                                    text-white transition-all duration-300
                                    group-hover:-translate-y-0.5 border border-white/10 hover:border-white/20
                                    hover:shadow-md"
                                >
                                    <span className="transition-opacity">
                                        Learn More
                                    </span>
                                </Button>
                            </Link>
                        </div>
                    </motion.div>

                    {/* Feature Cards */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 1.6, duration: 0.8 }}
                        className="mt-20 grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
                    >
                        {[
                            {
                                icon: BarChart3,
                                title: "Data Import",
                                desc: "Import historical NBA game data",
                            },
                            {
                                icon: BrainCircuit,
                                title: "ML Models",
                                desc: "Train prediction models",
                            },
                            {
                                icon: TrendingUp,
                                title: "Analysis",
                                desc: "Compare performance metrics",
                            },
                        ].map((feature, i) => {
                            const Icon = feature.icon;
                            return (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 1.8 + i * 0.1 }}
                                    className="backdrop-blur-md bg-white/5 hover:bg-white/10
                                    rounded-xl p-6 border border-white/10 hover:border-orange-500/50
                                    transition-all duration-300 hover:-translate-y-1 hover:shadow-lg hover:shadow-orange-500/20"
                                >
                                    <Icon className="w-12 h-12 mb-4 text-orange-500" strokeWidth={1.5} />
                                    <h3 className="text-xl font-semibold text-white mb-2">
                                        {feature.title}
                                    </h3>
                                    <p className="text-gray-400">{feature.desc}</p>
                                </motion.div>
                            );
                        })}
                    </motion.div>
                </motion.div>
            </div>
        </div>
    );
}
