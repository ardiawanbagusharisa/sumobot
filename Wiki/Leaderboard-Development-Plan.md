# Leaderboard Feature — Development Plan

**Project:** Sumobot (Unity, uGUI + TextMesh Pro, Newtonsoft JSON)
**Feature:** Leaderboards screen + ranking system
**Strategy:** Phase 1 — fully local leaderboard. Phase 2 — online backend for the AI competition.
**Author:** Drafted July 2026

---

## 1. Executive Summary

The main menu already contains a `Leaderboards` menu state, a `ButtonLeaderboard`, and an empty `MenuLeaderboards` GameObject stub in `MainMenu.unity` — but no leaderboard logic, UI content, ranking model, or persistence exists anywhere in the codebase. This plan describes how to build the feature end to end:

- **Phase 1 (local):** record every battle result at `PostBattle_ShowResult`, compute per-mode ratings (Elo), persist to `Application.persistentDataPath/Leaderboards/leaderboards.json`, and render the mockup UI (mode + control-mode filters, pinned own-rank, scrollable ranked list).
- **Phase 2 (online):** publish/fetch entries through a REST backend, integrate the existing `BattleSimulator` batch pipeline as the authoritative score source for the AI-script competition, and add submission validation and anti-cheat.

---

## 2. Current State Analysis

### 2.1 Mockup inventory

`Wiki/Figure/` contains 24 mockups (`1 splash.png` … `23 sfxandvfx.png` + `Git Flow.png`). **There is no dedicated leaderboard PNG in the repo** — the only leaderboard mockup is the one on the GitHub wiki (reproduced below). Related mockups that constrain this feature:

| Mockup | Relevance |
|---|---|
| `2 main menu.png` | Entry point; Leaderboards is reached from Main Menu (button already wired to `ShowLeaderboards()`) |
| `3 game mode.png` | Game modes: Campaign, Multiplayer |
| `5 player mode.png` | Player modes: PvP, PvAI, AIvAI → first dropdown of the leaderboard |
| `8 control mode.png` | Control modes: Buttons, Live Code (locked), AI Script (locked) → second dropdown |
| `11 post-battle.png` | Post-battle stats screen — the moment results are committed to the leaderboard |
| `13 gamestate.png` | Battle state flow feeding the result hook |

### 2.2 Leaderboard mockup breakdown (wiki image)

```
┌────────────────────────────────────────────────────────┐
│ [X]                                                     │
│                  L E A D E R B O A R D S               │
│                                                         │
│  [PvP ▾]   [Buttons ▾]              [#35  HarryPOTAH]  │
│  ┌──────────────────────────────────────────────┐  ▲   │
│  │ #1  ShyCoder15      BumbleWasp    👑   1332  │  █   │
│  │ #2  Jerry&TOM       GOBOT         👑   1215  │  █   │
│  │ #3  Ivan13          TWBot         👑   1147  │  ▼   │
│  │ #4  JustInNumberOne WorldChip     👑   1105  │      │
│  │ …                                            │      │
│  └──────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────┘
```

Elements the implementation must reproduce:

1. **Close (X) button**, top-left → `MainMenuController.ShowMainMenu()`.
2. **Title** "LEADERBOARDS".
3. **Two filter dropdowns** (TMP_Dropdown):
   - Player mode: `PvP`, `PvAI`, `AIvAI` (wiki: "3 player modes").
   - Control mode: `Buttons`, `Live Command`, `AI Script` (wiki §4: "three leaderboards … associated with the input types"). Maps 1:1 to `InputTypeExtension.ToBattleInputType()` categories (UI/Keyboard → 0, LiveCommand → 1, Script → 2).
4. **Pinned own-rank chip**, top-right: `#<rank> <playerName>` — always visible even when the player's row is scrolled out of view.
5. **Scrollable ranked list** (ScrollRect + vertical layout), each row: rank number, player name, **bot name** (second column — the bot/script used, cf. `Bot.ID` and README micro-competition table), crown icon, score/rating.
6. **Row highlight tiers:** gold row for #1, silver #2, bronze #3, default rows below.
7. **Scrollbar** on the right edge.

### 2.3 Code touchpoints (verified against source)

| File | What exists today | Leaderboard relevance |
|---|---|---|
| `Assets/Scripts/UI/MainMenuController.cs` | `MenuState.Leaderboards`, `ShowLeaderboards()` switch already present | Attach new leaderboard UI under the mapped `MenuLeaderboards` object |
| `Assets/Scenes/MainMenu.unity` | `ButtonLeaderboard` wired to `ShowLeaderboards`; `MenuLeaderboards` GameObject is an empty stub | Scene work happens here |
| `Assets/Scripts/Battle/Managers/BattleManager.cs` | `BattleState` machine; `PostBattle_ShowResult` state calls `LogManager.SortAndSave()`; `Battle` record exposes `GetBattleWinner()`, `LeftWinCount`, `RightWinCount`, `RoundSystem` | Single integration point to commit results |
| `Assets/Scripts/Battle/Managers/GameManager.cs` | `PlayerProfile { ID (Guid), Name, Parts, CurrentCostume }`; dummy profiles `Player1`/`Player2` created in `OnEnable`, **not persisted** | Needs persistent identity for ranking |
| `Assets/Scripts/Battle/Managers/LogManager.cs` | `PlayerStats { Name, InputType, SkillType, Bot, WinPerGame, WinPerRound, ActionTaken, ContactMade }`; writes JSON via Newtonsoft to `persistentDataPath/Logs/…` | Proven persistence pattern to copy; stats source |
| `Assets/Scripts/Battle/Input/InputProvider.cs` | `InputType { Keyboard, UI, LiveCommand, Script }` + `ToBattleInputType()` 3-way category mapping | Drives the control-mode dimension |
| `Assets/Scripts/Helper/BattleSimulator.cs` | Batch simulation with checkpoints, writes `Logs/Batch/<id>/game_*.json` | Phase 2: authoritative competition scoring |
| `Assets/Scripts/Bot/Core/Bot.cs` | `abstract string ID` per bot | Bot-name column |
| `Packages/manifest.json` | uGUI 2.0, TMP, Newtonsoft 3.2.2, **no networking/UGS packages**; `com.unity.modules.unitywebrequest` module enabled | Phase 2 uses `UnityWebRequest`; no server SDK today |
| `Assets/Scripts/Helper/EventRegistry.cs` / `TrackableEvent.cs` | String-keyed event registry used by `BattleManager.Events` | Reuse for `OnLeaderboardUpdated` |

### 2.4 Gaps

1. No persistent player identity (profiles are regenerated GUIDs each run).
2. No rating/ranking model anywhere.
3. No leaderboard persistence or service class.
4. `MenuLeaderboards` UI is empty.
5. `BattleManager` doesn't know the *player mode* (PvP/PvAI/AIvAI) as an explicit enum — it must be derived from the two `InputType`s or added as config.
6. No networking layer (Phase 2).

---

## 3. Requirements

### 3.1 Functional

- **FR-1** Record every completed battle (winner, loser, draw, per-side input type, bot ID, round system) exactly once, at `PostBattle_ShowResult`.
- **FR-2** Maintain a separate ranking table per **(player mode × control mode)** pair: 3 × 3 = 9 boards. (The mockup's two dropdowns select one board.)
- **FR-3** Rating: Elo (K=32, start 1000, floor 100). Draws count as 0.5. Win-rate, games played, and win streak stored alongside.
- **FR-4** UI per mockup §2.2: filters, pinned own rank, scrollable list, top-3 tier colors, crown icons, close button.
- **FR-5** Show placeholder state ("No battles recorded yet — go fight!") when a board is empty.
- **FR-6** Persist across sessions; corrupt/missing file must self-heal (recreate empty, back up corrupt file).
- **FR-7** Persistent local player profile (name editable later via Settings/Market; default `Player1`).
- **FR-8 (P2)** Fetch/submit entries to a remote backend; merge local+remote views; offline fallback to cached remote snapshot.
- **FR-9 (P2)** Competition ingestion: `BattleSimulator` batch results uploadable as authoritative AI-script board.

### 3.2 Non-functional

- **NFR-1** Zero GC-heavy work per frame: list virtualization or object pooling if > 100 rows.
- **NFR-2** Leaderboard I/O must never block the battle loop — write asynchronously (thread pool / `Task.Run`) or at scene transition.
- **NFR-3** All new code namespaced (`SumoLeaderboard`) matching existing conventions (`SumoManager`, `SumoInput`, `SumoBot`, …).
- **NFR-4** No new package dependencies in Phase 1.
- **NFR-5** Phase 2 requests must be resilient: timeout 10 s, retry ×2 with backoff, no gameplay stalls.
- **NFR-6** Local file tamper-evidence (HMAC of payload) in Phase 1; server-side validation in Phase 2.

---

## 4. Data Model (Phase 1)

New folder: `Assets/Scripts/Leaderboard/`

```csharp
namespace SumoLeaderboard
{
    public enum PlayerMode { PvP, PvAI, AIvAI }        // dropdown 1
    public enum ControlCategory { Buttons, LiveCommand, AIScript } // dropdown 2
                                                        // == ToBattleInputType() 0/1/2

    [Serializable]
    public class LeaderboardEntry
    {
        public string ProfileID;      // persistent GUID
        public string PlayerName;     // e.g. "HarryPOTAH"
        public string BotName;        // Bot.ID or "-" for human buttons play
        public int    Rating;         // Elo, start 1000
        public int    GamesPlayed;
        public int    Wins;
        public int    Losses;
        public int    Draws;
        public int    BestStreak;
        public int    CurrentStreak;
        public string UpdatedAtUtc;   // ISO-8601
    }

    [Serializable]
    public class LeaderboardTable
    {
        public PlayerMode Mode;
        public ControlCategory Control;
        public List<LeaderboardEntry> Entries = new();
    }

    [Serializable]
    public class LeaderboardData          // root object of leaderboards.json
    {
        public int Version = 1;           // schema migrations
        public string LocalProfileID;
        public string LocalPlayerName;
        public List<LeaderboardTable> Tables = new();  // ≤ 9
        public string Checksum;           // HMAC-SHA256 of canonical payload
    }

    public struct BattleResultRecord      // handed over by BattleManager
    {
        public PlayerMode Mode;
        public ControlCategory LeftControl, RightControl;
        public string LeftProfileID, RightProfileID;
        public string LeftName, RightName;
        public string LeftBot, RightBot;  // null for human sides
        public BattleWinner Winner;       // existing SumoManager enum
        public RoundSystem RoundSystem;
        public int LeftRoundWins, RightRoundWins;
    }
}
```

**Persistence path:** `Application.persistentDataPath/Leaderboards/leaderboards.json` (Newtonsoft, `Formatting.Indented`), mirroring `LogManager`'s pattern. Write via temp-file + `File.Replace` for atomicity. Corrupt file → renamed `leaderboards.corrupt.<timestamp>.json`, fresh file created (FR-6).

**Player mode derivation** (no new battle config needed):

```csharp
PlayerMode Derive(InputType left, InputType right)
{
    bool leftBot  = left  == InputType.Script;
    bool rightBot = right == InputType.Script;
    if (leftBot && rightBot) return PlayerMode.AIvAI;
    if (leftBot || rightBot) return PlayerMode.PvAI;
    return PlayerMode.PvP;
}
```

*(Design note: a human using AI-assist doesn't exist today; if it ever does, revisit.)*

---

## 5. Ranking Algorithm

Standard Elo per table:

```
expected = 1 / (1 + 10^((Rb - Ra)/400))
Ra' = Ra + K * (score - expected)     // score: win 1, draw 0.5, loss 0
K = 32 ;  floor = 100 ;  initial = 1000
```

- Both sides of a battle get updated in the **same** table (the battle's mode × each side's control category; if the two sides used different control categories — e.g. Buttons vs Script in PvAI — each side is scored on **its own** control board against the opponent's current rating).
- Draws (`BattleWinner.Draw`) apply 0.5/0.5.
- Rank = index after sorting by `Rating desc, Wins desc, GamesPlayed asc, PlayerName asc` (deterministic ties).
- Rationale for Elo over raw win-rate: matches the mockup's 4-digit scores (1332, 1215, …), robust to unequal game counts, and consistent with the micro-competition's competitive framing. Win-rate is still displayed in a row tooltip/detail later if desired.

---

## 6. Phase 1 — Local Leaderboard (detailed)

### 6.1 New files

```
Assets/Scripts/Leaderboard/
├── LeaderboardModels.cs        (§4 types)
├── EloCalculator.cs            (pure static, unit-testable)
├── LeaderboardStore.cs         (load/save/atomic-write/checksum/migration)
├── LeaderboardService.cs       (singleton MonoBehaviour, DontDestroyOnLoad;
│                                RecordBattle(BattleResultRecord),
│                                GetTable(mode, control) → sorted snapshot,
│                                GetLocalRank(mode, control),
│                                EventRegistry: "OnLeaderboardUpdated")
└── UI/
    ├── LeaderboardPanelController.cs   (dropdowns, refresh, pinned chip, empty state)
    ├── LeaderboardRowView.cs           (rank, name, bot, crown, score; tier styling)
    └── LeaderboardRowPool.cs           (simple pooling; virtualization if needed)
Assets/Prefabs/UI/LeaderboardRow.prefab
```

### 6.2 Modified files

| File | Change |
|---|---|
| `BattleManager.cs` | In the `PostBattle_ShowResult` case (next to the existing `LogManager.SortAndSave()`): build `BattleResultRecord` from `Battle`, `LeftInputType`, `RightInputType`, `GameManager.Instance.Left/Right`, `BotManager` and call `LeaderboardService.Instance.RecordBattle(record)`. Guard with an `alreadyRecorded` flag so a rematch/replay can't double-count. **Skip recording when `BattleSimulator` is enabled** (simulation ≠ ladder play). |
| `GameManager.cs` | Persist `PlayerProfile.ID`/`Name` (`PlayerPrefs` or a small `profile.json`) instead of regenerating GUIDs each run; expose `LocalProfile`. Right-side human keeps a stable second profile (`Player2`). |
| `MainMenuController.cs` | No code change required (state exists); optionally call `LeaderboardPanelController.Refresh()` from `ShowLeaderboards()` so data is fresh on open. |
| `MainMenu.unity` | Build the panel under `MenuLeaderboards` (§6.3). |

### 6.3 Scene / UI construction (uGUI + TMP, matching mockup)

Hierarchy under `MenuLeaderboards`:

```
MenuLeaderboards
├── Background (existing blue panel style, cf. other menu screens)
├── ButtonClose "X"          → MainMenuController.ShowMainMenu()
├── TitleText  "LEADERBOARDS" (TMP, letter-spaced, white — visual style of wiki)
├── FilterBar
│   ├── DropdownPlayerMode   (TMP_Dropdown: PvP / PvAI / AIvAI)
│   └── DropdownControlMode  (TMP_Dropdown: Buttons / Live Command / AI Script)
├── OwnRankChip              ("#35  HarryPOTAH", dark pill, top-right)
├── ScrollView (ScrollRect, vertical, Scrollbar right)
│   └── Content (VerticalLayoutGroup + ContentSizeFitter)
│       └── LeaderboardRow ×N (pooled prefab)
└── EmptyStateText           ("No battles recorded yet…")
```

Row prefab layout (HorizontalLayoutGroup): `RankText (10%) | PlayerName (32%) | BotName (28%) | CrownIcon (10%) | ScoreText (14%) | spacing`. Tier colors: `#F5C542` (#1), `#C9C9C9` (#2), `#C77B3F` (#3), default light row — matching the mockup's gold/grey/bronze rows. Crown sprite: reuse the existing crown asset (`BattleUIManager.CrownSprite` already references one). `TMP_Dropdown` is already the established pattern (`BattleUIManager.LeftInputType`).

Behaviors:

- Dropdown change → `LeaderboardService.GetTable(mode, control)` → repopulate pool; play `SFXManager.Instance.Play2D("ui_accept")` for consistency with `MainMenuController`.
- Own-rank chip: `GetLocalRank(...)`; hide when local profile has no entry on the selected board.
- Locked control modes (Live Code / AI Script are campaign-locked per `8 control mode.png`): the dropdown still lists them — leaderboards are viewable even if the mode is locked for play.

### 6.4 Edge cases

1. Double-commit on rematch → `alreadyRecorded` reset only in `Battle_Preparing`.
2. Draws → 0.5 Elo, `Draws++`, streaks reset.
3. Battle quit mid-way (scene unload before `PostBattle_ShowResult`) → nothing recorded, by design.
4. Renamed player → entries keyed by `ProfileID`; `PlayerName` refreshed on every record.
5. Same profile on both sides (local hot-seat vs self) → record once per side but flag `selfPlay`; optionally exclude from board to prevent farming (config bool, default: excluded for PvP, included for AIvAI since AIvAI is legitimate bot testing).
6. Simulation batches (`BattleSimulator`) never touch the local ladder (Phase 1) — they are Phase 2's competition input.
7. File corruption / partial write → checksum mismatch → back up + reset (FR-6).

### 6.5 Testing (com.unity.test-framework already in manifest)

- **EditMode:** `EloCalculatorTests` (symmetry, draw, floor, K), `LeaderboardStoreTests` (roundtrip, corrupt-file recovery, migration stub, atomic write), sorting/tie-break determinism, `Derive(PlayerMode)` truth table (all 4×4 InputType pairs).
- **PlayMode:** scripted `BestOf1` battle → assert exactly one `RecordBattle`; rematch double-count guard; UI smoke test — populate 50 fake entries, assert row count, tier colors, own-rank chip text.
- **Manual QA checklist:** all 9 board combos, empty boards, 200+ entries scroll perf, name with rich-text chars (sanitize `<` `>`), quit-mid-battle.

### 6.6 Phase 1 work breakdown & estimates

| # | Task | Est. |
|---|---|---|
| 1 | `LeaderboardModels` + `EloCalculator` + unit tests | 0.5 d |
| 2 | `LeaderboardStore` (atomic IO, checksum, recovery) + tests | 1 d |
| 3 | `LeaderboardService` + `BattleManager`/`GameManager` integration | 1 d |
| 4 | Profile persistence | 0.5 d |
| 5 | UI prefab + panel controller + pooling | 2 d |
| 6 | Scene wiring, SFX, polish (tier colors, crown, empty state) | 1 d |
| 7 | PlayMode tests + manual QA + fixes | 1 d |
| | **Total Phase 1** | **≈ 7 dev-days** |

---

## 7. Phase 2 — Online Leaderboard & Competition

### 7.1 Backend options (decision needed — ADR recommended)

| Option | Pros | Cons |
|---|---|---|
| **A. Unity Gaming Services Leaderboards + Auth (anonymous)** | Managed, free tier, SDK handles auth/retries; fastest to ship | Less control over custom keys (9 boards → 9 UGS leaderboards, fine); vendor lock-in; limited server-side validation hooks |
| **B. Custom REST service (e.g. FastAPI/Express + Postgres, hosted small)** | Full control: competition ingestion, LLM-assisted script validation pipeline (wiki §4), custom anti-cheat | You own uptime, auth, cost |
| **C. GitHub-as-backend (scores committed to repo/wiki by CI running `BattleSimulator`)** | Matches current competition workflow (PR-based submissions, README table); zero server | Not real-time; only fits the AI-script board |

**Recommendation:** A for the in-game ladders (PvP/PvAI casual boards), **C→B migration path** for the AI-script competition board: keep the existing PR + simulation workflow, have CI emit a `competition_leaderboard.json` artifact the game fetches raw from GitHub (`UnityWebRequest`, module already enabled), then graduate to B when submission volume justifies it.

### 7.2 Client architecture changes

- `ILeaderboardBackend` interface with `LocalBackend` (Phase 1 store), `UgsBackend`, `StaticJsonBackend` (raw GitHub JSON).
- `LeaderboardService` composes backends per board: competition board = remote-only w/ cached snapshot (`persistentDataPath/Leaderboards/cache/…`); casual boards = local now, UGS later.
- UI additions: source indicator (Local / Global), last-sync timestamp, pull-to-refresh button, loading + offline states.

### 7.3 API sketch (if/when option B)

```
POST /v1/matches            {payload, hmac}     → 202  (rating computed server-side)
GET  /v1/leaderboards/{mode}/{control}?offset&limit → entries + requester rank
GET  /v1/players/{id}                            → profile + per-board stats
POST /v1/competition/submissions                 → bot script metadata (PR URL)
```

Anti-cheat baseline: server-computed Elo only (client never posts ratings); per-device anonymous auth token; rate limiting; match payload includes round logs digest so suspicious results (e.g. 1000 wins/hour) are flaggable; competition board immune by design (scores only from the official simulation run).

### 7.4 Phase 2 breakdown (high-level)

| # | Task | Est. |
|---|---|---|
| 1 | ADR: backend choice (use `engineering:architecture` template) | 0.5 d |
| 2 | `ILeaderboardBackend` refactor + cached `StaticJsonBackend` | 1.5 d |
| 3 | CI: `BattleSimulator` headless run → `competition_leaderboard.json` artifact | 2 d |
| 4 | UGS integration (auth + 9 boards) *or* custom API client | 3–5 d |
| 5 | UI: source toggle, sync states, error/offline UX | 1.5 d |
| 6 | Security pass + load/edge testing | 1.5 d |
| | **Total Phase 2** | **≈ 10–12 dev-days** |

---

## 8. Milestones

1. **M1 (end of week 1):** Elo + store + service merged; battles recorded locally; verifiable via JSON file.
2. **M2 (mid week 2):** Leaderboards screen fully functional per mockup; Phase 1 QA done. *Ship in next build.*
3. **M3:** Backend ADR accepted; competition board reading CI-generated JSON.
4. **M4:** Global casual boards live (UGS or custom); anti-cheat baseline in place.

## 9. Risks & Open Questions

- **R1:** `GameManager` is marked "Temporary"/dummy profiles — profile persistence (task 4, §6.2) is a prerequisite; coordinate with whoever owns the planned Settings/Market screens (name entry UI lives there eventually; interim: default names).
- **R2:** Player mode derivation assumes `InputType.Script` ⇔ AI side; confirm intended classification of `LiveCommand` (currently treated as human control — consistent with wiki §3.2).
- **R3:** Mockup shows a single score column; product should confirm Elo vs XP/coins display (wiki says winnings convert to XP/coins — that is the Market economy, kept separate from ranking here).
- **R4:** Local-only boards can be farmed vs weak bots; acceptable for Phase 1 (single device), solved server-side in Phase 2.
- **Q1:** Should Campaign results feed any board? (Current answer: no — boards are Multiplayer/competition only.)
- **Q2:** Row count cap per board? (Suggest keep top 500 + always retain local profile entries.)
