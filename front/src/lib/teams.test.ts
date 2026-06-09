import { describe, expect, it } from "vitest";
import { TEAMS, TEAM_BY_ID, getTeamAbbr, getTeamName } from "./teams";

describe("teams library", () => {
  it("exposes all 30 NBA franchises", () => {
    expect(TEAMS).toHaveLength(30);
  });

  it("uses unique IDs in the standard nba.com range", () => {
    const ids = TEAMS.map((t) => t.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const id of ids) {
      expect(id).toBeGreaterThanOrEqual(1610612737);
      expect(id).toBeLessThanOrEqual(1610612766);
    }
  });

  it("uses unique 3-letter abbreviations", () => {
    const abbrs = TEAMS.map((t) => t.abbr);
    expect(new Set(abbrs).size).toBe(abbrs.length);
    for (const a of abbrs) {
      expect(a).toMatch(/^[A-Z]{3}$/);
    }
  });

  it("TEAM_BY_ID is keyed by string id", () => {
    expect(TEAM_BY_ID["1610612738"].abbr).toBe("BOS");
    expect(TEAM_BY_ID["1610612747"].name).toBe("Los Angeles Lakers");
  });

  describe("getTeamName", () => {
    it("returns the display name for a known id (string or number)", () => {
      expect(getTeamName(1610612738)).toBe("Boston Celtics");
      expect(getTeamName("1610612738")).toBe("Boston Celtics");
    });

    it("falls back to the raw id when unknown", () => {
      expect(getTeamName("9999999999")).toBe("9999999999");
      expect(getTeamName(0)).toBe("0");
    });
  });

  describe("getTeamAbbr", () => {
    it("returns 3-letter abbreviation for a known id", () => {
      expect(getTeamAbbr(1610612744)).toBe("GSW");
    });

    it("falls back to raw id for unknown id", () => {
      expect(getTeamAbbr("nope")).toBe("nope");
    });
  });
});
