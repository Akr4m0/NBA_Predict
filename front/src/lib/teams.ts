// NBA team ID → display info.
// IDs follow the standard nba.com convention (1610612737–1610612766).

export interface NbaTeam {
  id: number;
  name: string;
  abbr: string;
}

export const TEAMS: readonly NbaTeam[] = [
  { id: 1610612737, name: "Atlanta Hawks",           abbr: "ATL" },
  { id: 1610612738, name: "Boston Celtics",          abbr: "BOS" },
  { id: 1610612739, name: "Cleveland Cavaliers",     abbr: "CLE" },
  { id: 1610612740, name: "New Orleans Pelicans",    abbr: "NOP" },
  { id: 1610612741, name: "Chicago Bulls",           abbr: "CHI" },
  { id: 1610612742, name: "Dallas Mavericks",        abbr: "DAL" },
  { id: 1610612743, name: "Denver Nuggets",          abbr: "DEN" },
  { id: 1610612744, name: "Golden State Warriors",   abbr: "GSW" },
  { id: 1610612745, name: "Houston Rockets",         abbr: "HOU" },
  { id: 1610612746, name: "LA Clippers",             abbr: "LAC" },
  { id: 1610612747, name: "Los Angeles Lakers",      abbr: "LAL" },
  { id: 1610612748, name: "Miami Heat",              abbr: "MIA" },
  { id: 1610612749, name: "Milwaukee Bucks",         abbr: "MIL" },
  { id: 1610612750, name: "Minnesota Timberwolves",  abbr: "MIN" },
  { id: 1610612751, name: "Brooklyn Nets",           abbr: "BKN" },
  { id: 1610612752, name: "New York Knicks",         abbr: "NYK" },
  { id: 1610612753, name: "Orlando Magic",           abbr: "ORL" },
  { id: 1610612754, name: "Indiana Pacers",          abbr: "IND" },
  { id: 1610612755, name: "Philadelphia 76ers",      abbr: "PHI" },
  { id: 1610612756, name: "Phoenix Suns",            abbr: "PHX" },
  { id: 1610612757, name: "Portland Trail Blazers",  abbr: "POR" },
  { id: 1610612758, name: "Sacramento Kings",        abbr: "SAC" },
  { id: 1610612759, name: "San Antonio Spurs",       abbr: "SAS" },
  { id: 1610612760, name: "Oklahoma City Thunder",   abbr: "OKC" },
  { id: 1610612761, name: "Toronto Raptors",         abbr: "TOR" },
  { id: 1610612762, name: "Utah Jazz",               abbr: "UTA" },
  { id: 1610612763, name: "Memphis Grizzlies",       abbr: "MEM" },
  { id: 1610612764, name: "Washington Wizards",      abbr: "WAS" },
  { id: 1610612765, name: "Detroit Pistons",         abbr: "DET" },
  { id: 1610612766, name: "Charlotte Hornets",       abbr: "CHA" },
];

export const TEAM_BY_ID: Record<string, NbaTeam> = Object.fromEntries(
  TEAMS.map((t) => [String(t.id), t]),
);

/**
 * Returns the team's display name for a given ID, falling back to the raw ID
 * if the team isn't in our map (e.g. historical relocations, future expansion).
 */
export function getTeamName(id: string | number): string {
  return TEAM_BY_ID[String(id)]?.name ?? String(id);
}

/**
 * Returns the 3-letter abbreviation (e.g. "BOS"), falling back to the raw ID.
 */
export function getTeamAbbr(id: string | number): string {
  return TEAM_BY_ID[String(id)]?.abbr ?? String(id);
}
