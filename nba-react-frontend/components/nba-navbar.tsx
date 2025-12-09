"use client";

import {
    Home,
    Upload,
    BrainCircuit,
    BarChart3,
    LayoutDashboard,
    CheckCircle2,
} from "lucide-react";
import { NavBar } from "@/components/ui/tubelight-navbar";

export function NBANavBar() {
    const navItems = [
        { name: "Home", url: "/", icon: Home },
        { name: "Import", url: "/import", icon: Upload },
        { name: "Train", url: "/train", icon: BrainCircuit },
        { name: "Analysis", url: "/analysis", icon: BarChart3 },
        { name: "Dashboard", url: "/dashboard", icon: LayoutDashboard },
        { name: "Verify", url: "/verify", icon: CheckCircle2 },
    ];

    return <NavBar items={navItems} />;
}
