/**
 * Global ambient backdrop rendered once in App.tsx. Fixed + pointer-events-none
 * so it never intercepts clicks and sits behind all content. Provides the base
 * navy (so page roots can be transparent), a faint grid for depth, and soft
 * orange/blue glow blobs using the existing theme tokens.
 */
export function AppBackground() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-background"
    >
      {/* faint grid */}
      <div
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage:
            "linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)",
          backgroundSize: "52px 52px",
          maskImage: "radial-gradient(ellipse 80% 60% at 50% 0%, black 40%, transparent 100%)",
          WebkitMaskImage:
            "radial-gradient(ellipse 80% 60% at 50% 0%, black 40%, transparent 100%)",
        }}
      />
      {/* ambient glow blobs */}
      <div className="absolute -left-32 -top-40 h-[40rem] w-[40rem] rounded-full bg-primary/10 blur-[140px]" />
      <div className="absolute -right-40 top-1/3 h-[36rem] w-[36rem] rounded-full bg-secondary/10 blur-[150px]" />
      <div className="absolute bottom-0 left-1/3 h-[32rem] w-[32rem] rounded-full bg-primary/[0.06] blur-[160px]" />
    </div>
  );
}
