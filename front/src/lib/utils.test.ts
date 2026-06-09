import { describe, expect, it } from "vitest";
import { cn } from "./utils";

describe("cn (className merger)", () => {
  it("joins string args with a space", () => {
    expect(cn("a", "b")).toBe("a b");
  });

  it("filters out falsy values", () => {
    expect(cn("a", null, undefined, false, "b")).toBe("a b");
  });

  it("respects clsx object/array notation", () => {
    expect(cn(["a", { b: true, c: false }])).toBe("a b");
  });

  it("dedupes conflicting tailwind classes via tailwind-merge", () => {
    // Later class wins for the same utility group.
    expect(cn("p-2", "p-4")).toBe("p-4");
    expect(cn("text-red-500", "text-blue-500")).toBe("text-blue-500");
  });

  it("keeps non-conflicting tailwind classes together", () => {
    const out = cn("p-2", "text-red-500", "rounded-lg");
    expect(out.split(" ").sort()).toEqual(["p-2", "rounded-lg", "text-red-500"]);
  });
});
