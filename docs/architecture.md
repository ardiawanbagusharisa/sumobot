# Sumobot — System Architecture Overview

> Living document. Keep it short and current; when a boundary changes, edit the
> relevant section rather than appending notes. For the *why* behind hard-to-reverse
> calls, write a Backlog decision (`.backlog/decisions/`) and link it here.

Unity 2D top-down sumo robot battle game + AI competition platform. Players learn
programming/AI by writing bot scripts, then battle (human-vs-human, human-vs-bot,
or bot-vs-bot) and train/evaluate bots in bulk.

---

## 1. Scenes

All scenes live in `Assets/Scenes/` and are registered in Build Settings.

| Scene | Role | Entered from |
|-------|------|--------------|
| **MainMenu** | Shell for all menus. Sub-"pages" (Login, GameModes, Campaign, Multiplayer, Store, Garage, Leaderboards, Settings) are **not** separate scenes — they are GameObjects toggled by `MainMenuController.SwitchMenu`. | Startup |
| **Battle** | Core battle scene. **Currently doubles as the training/simulation scene** (see §3). Multiplayer > Local loads this. | `MainMenuController.GoToBattle()` → `LoadScene("Battle")` |
| **BotCreator** | Per-side costume/part editor (`PartSwitcher` / `CostumeManager`), despite the name — edits the loadout of the profile it was opened for. Bot *scripts* are authored as files under `Assets/Resources/Bot/`, not in this scene. | From the **Battle** scene: a customize button (`BattleUIManager` → `GameManager.Battle_LoadCostumeScene(id)`); `BotCreator_SaveAndBack()` returns to Battle. |
| **Replay** | Play back recorded matches (`ReplayManager`). | From the **Battle** scene (`BattleUIManager` → `GameManager.Battle_ShowReplay()`); `Replay_BackToBattle()` returns. |
| **Campaign_1_1 / 2_1 / 3_1** | Level variants of the Battle scene, each with a `CampaignLevelController` that sets the level's player/opponent bot and control mode without editing the shared battle rules. | Campaign panel buttons |

---

## 2. Battle scene composition

The Battle scene root carries the manager stack (one GameObject, MonoBehaviours):

| Manager | Responsibility |
|---------|----------------|
| `BattleManager` | Battle state machine: `PreBattle_Preparing → Battle_Preparing → Countdown → Ongoing → End → Reset → PostBattle_ShowResult`. Owns config (round system, time, action interval, arena). |
| `BotManager` | Instantiates/updates the Left & Right `Bot` instances; assign/swap/uninit; routes `OnBotUpdate` / bounce / state-change callbacks. |
| `PacingManager` | Push/pacing constraints. |
| `BattleUIManager` | HUD. |
| `LogManager` / `ReplayManager` | Match logging + replay recording. |
| `BGMManager` / `SFXManager` / `VFXManager` | Presentation. |
| `BattleSimulator` | **Training/evaluation harness** (batch of AI-vs-AI configs). See §3. |

Bot layer (shared by play and training):
- `Assets/Scripts/Bot/Core/` — `Bot` base, `SumoAPI`, `BotUtility`.
- `Assets/Resources/Bot/Example/` — sample bots: RuleBased (FSM/BT/Fuzzy/Utility), ML (NN/DQN/PPO/Classification/LLM/SLM), EA (GA/MCTS), Micro-Competition entries.

---

## 3. Play vs Train — current coupling (the important part)

**Play and Train share one Battle scene.** The switch is a single Inspector
checkbox: whether the `BattleSimulator` component is enabled.

`BattleManager.OnEnable()`:

```csharp
if (simulator.enabled)                 // ← decides the scene's identity
{
    if (simulator.Mode == Simple) Init();
    simulator.PrepareSimulation();     // TRAIN path
}
else
    Init();                            // PLAY path
```

| Aspect | Play | Train (`BattleSimulator`) |
|--------|------|---------------------------|
| Who controls sides | `BattleManager.Left/RightInputType` (UI / Keyboard / LiveCommand / Script) + `BotManager.Left/RightEnabled` | `SelectedAgents[]` paired up, pushed into `BotManager` via `ApplyConfig` |
| Match count | 1 match (BestOf N rounds) | `TotalSimulations` (Simple) or N configs × Iteration (Advanced) |
| Speed | `timeScale = 1` | `SimpleTimeScale` / `ConfigTimeScale` (accelerated) |
| Presentation | HUD/BGM/SFX/VFX on | `PrepareSimulation` disables BGM/SFX/VFX |
| End behaviour | Show result, allow rematch | Write checkpoint, then `EditorApplication.isPlaying = false` (or `Application.Quit`) |
| Persistence | none | `simulation_checkpoint.json` under `Application.persistentDataPath/Settings/` |

**Known sharp edge:** because the mode is "guess from `simulator.enabled`", leaving the
component enabled makes Multiplayer > Local silently run the training batch instead of a
match. When the checkpoint is fully consumed the batch completes instantly and exits Play,
which reads as a bug. (Fixed short-term by disabling the component — "Plan A".)

Batch/offline training tooling also lives in `Simulation/` and
`Assets/Scripts/Editor/BattleSimulatorEditor.cs`, i.e. the original intent was to move
training toward tools/CLI, not a player-facing button.

---

## 4. Services layer

Backend-agnostic seam (see `CLAUDE.md` for the full contract):
- Composition root: `GameServices` (`Assets/Scripts/Services/Core/`). Swap Local* stubs
  for Photon/UGS **only here**.
- Surfaces: `IAuthService`, `IPlayerDataService`, `ICatalogService`, `ILeaderboardService`
  (`ILeaderboardService` lives in `Assets/Scripts/Leaderboard/`).
- `GameManager` boots services + session; keeps scene routing/profiles, not backend wiring.

---

## 5. Proposed play/train split (roadmap)

Recorded here as direction; promote to a Backlog decision when acted on.

- **Plan A — done/short-term:** disable `BattleSimulator` in `Battle.unity` so Local is a
  real match again.
- **Plan B — explicit mode flag:** replace "guess from `simulator.enabled`" with an
  explicit `BattleMode { Play, Train }` set by the caller entering the scene; `OnEnable`
  switches on it.
- **Plan C — scene-level split:** dedicated `Simulation.unity` for training (no HUD/audio),
  `Battle.unity` for play only; shared logic stays in shared scripts; training entry moves
  to Editor tool / CLI batch (training calls `Application.Quit`, so it must not be a normal
  menu button).
