"use client";

import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { usePathname } from "next/navigation";

interface NavItem {
    name: string;
    url: string;
    icon: LucideIcon;
}

interface NavBarProps {
    items: NavItem[];
    className?: string;
}

export function NavBar({ items, className }: NavBarProps) {
    const pathname = usePathname();
    const [activeTab, setActiveTab] = useState(items[0].name);
    const [isMobile, setIsMobile] = useState(false);

    useEffect(() => {
        // Set active tab based on current pathname
        const currentItem = items.find((item) => item.url === pathname);
        if (currentItem) {
            setActiveTab(currentItem.name);
        }
    }, [pathname, items]);

    useEffect(() => {
        const handleResize = () => {
            setIsMobile(window.innerWidth < 768);
        };

        handleResize();
        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    }, []);

    return (
        <div
            className={cn(
                "fixed bottom-6 sm:top-6 left-1/2 -translate-x-1/2 z-50",
                className
            )}
        >
            <div className="flex items-center gap-2 bg-[#0a0f1c]/95 border border-white/10 backdrop-blur-xl py-2 px-2 rounded-full shadow-2xl shadow-orange-900/30">
                {items.map((item) => {
                    const Icon = item.icon;
                    const isActive = activeTab === item.name;

                    return (
                        <Link
                            key={item.name}
                            href={item.url}
                            onClick={() => setActiveTab(item.name)}
                            className={cn(
                                "relative cursor-pointer text-sm font-semibold px-5 py-2.5 rounded-full transition-all duration-300",
                                "flex items-center justify-center min-w-[44px]",
                                isActive
                                    ? "text-white"
                                    : "text-slate-400 hover:text-slate-200"
                            )}
                        >
                            {/* Text for desktop - with proper z-index to prevent interlacing */}
                            <span
                                className={cn(
                                    "hidden md:inline relative z-20",
                                    "transition-all duration-300"
                                )}
                                style={{
                                    textShadow: isActive
                                        ? "0 0 10px rgba(249, 115, 22, 0.5)"
                                        : "none",
                                }}
                            >
                                {item.name}
                            </span>

                            {/* Icon for mobile - with proper z-index and display */}
                            <span
                                className={cn(
                                    "md:hidden relative z-20 flex items-center justify-center",
                                    "transition-all duration-300"
                                )}
                            >
                                <Icon
                                    size={20}
                                    strokeWidth={2.5}
                                    className={cn(
                                        "transition-all duration-300",
                                        isActive && "drop-shadow-[0_0_8px_rgba(249,115,22,0.8)]"
                                    )}
                                />
                            </span>

                            {/* Active indicator with tubelight effect */}
                            {isActive && (
                                <motion.div
                                    layoutId="lamp"
                                    className="absolute inset-0 w-full bg-orange-600/20 rounded-full -z-0"
                                    initial={false}
                                    transition={{
                                        type: "spring",
                                        stiffness: 350,
                                        damping: 35,
                                    }}
                                >
                                    {/* Tubelight glow on top */}
                                    <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-10 h-1.5 bg-gradient-to-b from-orange-500 to-orange-600 rounded-t-full shadow-lg shadow-orange-500/50">
                                        {/* Glow effects */}
                                        <div className="absolute w-14 h-8 bg-orange-500/30 rounded-full blur-xl -top-3 -left-2 animate-pulse" />
                                        <div className="absolute w-10 h-6 bg-orange-400/40 rounded-full blur-lg -top-2 left-0" />
                                        <div className="absolute w-6 h-5 bg-orange-300/50 rounded-full blur-md -top-1 left-2" />
                                    </div>

                                    {/* Additional ambient glow */}
                                    <div className="absolute inset-0 bg-orange-500/10 rounded-full blur-sm" />
                                </motion.div>
                            )}
                        </Link>
                    );
                })}
            </div>
        </div>
    );
}
