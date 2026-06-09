import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useIsFetching } from "@tanstack/react-query";

export function TopLoadingBar() {
  const isFetching = useIsFetching();
  const [phase, setPhase] = useState<"hidden" | "loading" | "completing">("hidden");

  useEffect(() => {
    if (isFetching > 0) {
      setPhase("loading");
    } else if (phase === "loading") {
      setPhase("completing");
      const t = setTimeout(() => setPhase("hidden"), 300);
      return () => clearTimeout(t);
    }
  }, [isFetching]);

  return (
    <div className="pointer-events-none fixed left-0 right-0 top-0 z-[100] h-0.5">
      <AnimatePresence>
        {phase !== "hidden" && (
          <motion.div
            key="bar"
            className="h-full origin-left bg-orange-500"
            initial={{ width: "0%" }}
            animate={
              phase === "completing" ? { width: "100%" } : { width: "90%" }
            }
            exit={{ opacity: 0, transition: { duration: 0.2 } }}
            transition={
              phase === "completing"
                ? { duration: 0.15, ease: "easeOut" }
                : { duration: 0.8, ease: "easeOut" }
            }
          />
        )}
      </AnimatePresence>
    </div>
  );
}
