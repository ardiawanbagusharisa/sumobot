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

## 4. Bot appearance (costume) system

### Data model

- `CatalogItem` (`Assets/Scripts/Services/Catalog/`) — one purchasable/ownable thing:
  `Id`, `Slot`, `Price`, `IconResourcePath` (cropped market-grid icon), `IconColor` (tint for
  shared-sprite color variants). `SkinItem : CatalogItem` adds `PartSprite` — the *real*
  `Sprites/Character/...` source the bot renders in battle. `IconResourcePath` never renders
  on a bot; `PartSprite` never renders in the shop grid.
- `PlayerData` (`Assets/Scripts/Services/PlayerData/PlayerData.cs`) — the save: account-scoped
  `OwnedItemIds`/`Coins`, plus per-side costume via `Loadouts: List<Loadout>` (each a
  `Dictionary<slot, itemId>` called `EquippedBySlot`) and `ActiveLoadoutBySide: {Left/Right →
  loadoutId}`.
- **decision-5 invariant**: one account per device: inventory/coins are shared, but the
  *equipped set* is per local seat — Left and Right can wear different skins from the same
  owned pool by pointing at different `Loadout`s.

### Resolution (one shared implementation)

`EquippedCostumeResolver.ApplyEquipped(parts, tints, equippedBySlot)`
(`Assets/Scripts/Battle/Sumo/EquippedCostumeResolver.cs`) is the single place that turns a
loadout's `EquippedBySlot` into sprites: parses the slot string to `SumoPart`, looks up the
`SkinItem` in `GameServices.Catalog`, `Resources.Load`s `PartSprite`, and resolves the tint from
`IconColor`. Every call site below shares this function — no duplicate resolution logic.
Unequipped slots fall back to `DefaultCostume.SpriteFor(part)` (`Assets/Scripts/Battle/Sumo/
DefaultCostume.cs`) — the single source of the "naked" base look, shared by the battle
fallback, the Garage preview, and the Garage inventory's None/Unequip cell.

### Rendering

`SumoCostume` (`Assets/Scripts/Battle/Sumo/SumoCostume.cs`) has two modes on one component:
- `UI=false` (the real battle robot): resolves child `SpriteRenderer`s by tag (`Robot/{part}`);
  `AttachObject(parts, tints)` sets `.sprite` and `.color` directly.
- `UI=true` (HUD portrait): `AttachToUI` copies sprite/color/material from an already-dressed
  `SumoCostume` instance onto UI `Image`s.

The per-instance tint reaches material-based rendering via `DoodleDistort.shader`, which now
passes `SpriteRenderer.color` through as a vertex color and multiplies it in the fragment stage
(added alongside the tintable-skins feature).

### Where it's applied

| Site | Trigger | Which loadout |
|------|---------|----------------|
| Battle robot | `SumoController.Initialize` sets `Profile.Side`, calls `PlayerProfile.SetCostume` → `ApplyEquippedForLocalPlayer` | `PlayerData.GetLoadoutForSide(Side)` |
| Garage live preview | `GaragePreviewController.Refresh`, on `EquipmentChanged`; renders into a RenderTexture | `PlayerData.ActiveLoadout` (the one being edited) |
| BotCreator part switcher | `PartSwitcher` (◀▶ arrows), calls `EquipAsync`/`UnequipAsync`; opens by `GameManager.EditingID` | `SelectEditingSideLoadout()` points `ActiveLoadout` at the side BotCreator was opened for |
| Garage/BotCreator list icons | `GarageInventoryController` (owned-items list), `GarageLoadoutController` (per-slot tab preview) | `PlayerData.ActiveLoadout.EquippedBySlot` |

**Sharp edge already hit once:** `PartSwitcher` can repoint the service's shared
`ActiveLoadout` at the Right side while editing P2's bot in BotCreator. `GarageLoadoutController.
OnEnable` re-asserts the Left loadout on every open specifically to guard against that
selection leaking into Garage (Garage always edits the signed-in player's own gear).

---

## 5. Bot logic (module node-graph) system

Where §4 is what a bot *looks* like, this is what a bot *does*. Player-authored
behavior is a **module node-graph saved as data and interpreted at runtime** — no
free-form code, so no sandbox (**decision-8**). The graph flows straight into the
Publish payload (decision-7) as a `Source=Player` creation.

### Assembly boundary (the key seam)

| Layer | Assembly | Contents | May reference |
|-------|----------|----------|---------------|
| **Data** | `Sumobot.BotGraph` (`Assets/Scripts/BotGraph/`, ns `SumoBot.Graph`) | `BotGraph`/`GraphNode`/`GraphConnection` model, `ModuleLibrary` vocabulary, `GraphValidator`, `GraphSerializer`, `GraphInterpreter`, `IBotSensors`, `GraphActionIntent`; authoring layer (`Authoring/`, ns `SumoBot.Graph.Authoring`): `GraphDocument`, `GraphDraft`, `IDraftStore`/`FileDraftStore` | Newtonsoft + Logging — **no `UnityEngine`, no `SumoAPI`** (`FileDraftStore` takes an injected root dir instead of `Application.persistentDataPath`) |
| **Runtime** | game assembly (`Assets/Scripts/Bot/Core/`, ns `SumoBot`) | `GraphBot : Bot`, `SumoApiSensors` | the data layer + engine |

Dependency points **one way**: the runtime adapts the pure data layer, never the
reverse. That is why the whole interpreter is unit-tested without Unity (EditMode
tests under `BotGraph/Tests/`).

### Data model & vocabulary

- `BotGraph` = `Nodes[]` + `Connections[]` + `FormatVersion` (currently `1`). The
  version is explicit from day one because this shape persists into player saves and
  Market listings — **bump it on incompatible changes** (same discipline as decision-7).
- `GraphNode` names a `ModuleDefinition` by `TypeId` and carries numeric `Params`
  (bools are `0`/non-zero). `ModuleLibrary.BuildDefault()` is the **fixed MVP vocabulary**
  and the safe representation itself — a graph may only reference these TypeIds:

  | Kind | Modules |
  |------|---------|
  | Sensor | `enemyDistance`, `enemyAngle`, `edgeProximity`, `dashReady`, `skillReady`, `selfOutOfArena`, `enemyOutOfArena` |
  | Logic | `constant`, `greaterThan`, `lessThan`, `and`, `or`, `not` |
  | Action | `accelerate`, `turnLeft`, `turnRight`, `dash`, `skillBoost`, `skillStone` (fire while their `when` trigger is true) |

  Operators/directions/skills are distinct TypeIds so params stay purely numeric — no
  enum/choice params in the MVP. Ports are typed `Number`/`Bool`; connections are only
  legal between matching types.

### Validation → serialization → execution

- `GraphValidator.Validate` returns every `GraphError` (empty = valid): unknown module,
  duplicate node id, bad port/direction/type, fan-in > 1, unconnected required input, and
  **cycles** (data must flow acyclically sensors → logic → actions). Codes let the future
  editor highlight the offending node.
- `GraphSerializer` is the JSON round-trip (Newtonsoft).
- `GraphInterpreter.Evaluate(IBotSensors)` → `GraphActionIntent[]`. **Pure and
  deterministic**: output depends only on graph + sensor values; Action nodes are visited
  in graph order; outputs are memoized and cycle-guarded so even an unvalidated graph fails
  safe (no intents) rather than throwing. All port values are carried as `float`
  (Bool = 0/1, threshold 0.5).

### Runtime wiring

`GraphBot : Bot` holds the graph as a `[SerializeField] graphJson` string so it survives
the `Instantiate` clone that `BotManager.AssignGraphBot` performs per side. `Configure(graph)`
sets it; `OnBotInit` parses + validates **once** (missing/invalid → the bot idles, a
deliberate fail-safe); `OnBotUpdate` builds a `SumoApiSensors` snapshot each tick, runs the
interpreter, maps each `GraphActionIntent` to an `ISumoAction` via `ToAction`, and
`Enqueue`/`Submit`s to `SumoController`. (See the data-flow diagram accompanying decision-8.)

### Authoring (editor data bridge + drafts)

The `Authoring/` layer is the E3.1 foundation the Workshop editor (decision-10) sits on —
still pure data (no scene), so it is unit-tested the same way as the interpreter.

- `GraphDocument` wraps a `BotGraph` as the **single source of truth** for an editing
  session. Every gesture goes through it — `AddNode`/`RemoveNode` (removing a node also
  drops its incident connections), `MoveNode`, `SetParam`, `AddConnection`/`RemoveConnection` —
  and mutates that one model. `AddNode` only builds from the `ModuleLibrary` vocabulary and
  seeds parameter defaults; each visual node (E3.2) resolves back to its `GraphNode` via
  `TryGetNode(nodeId)`. `Validate()`/`IsValid` run `GraphValidator` live; `ToJson()` /
  `FromJson()` are the one-line save/load off the wrapped graph. The board stays **permissive**
  (an illegal wire is allowed and surfaced by the validator, not blocked at mutation time).
- `GraphDraft` = a stable `Id` + a `BotGraph`. Deliberately **not** a catalog item — no owner,
  price, or sellable flag; publishing (E5) is what freezes a draft into an owned item.
- `IDraftStore` is the persistence seam (mirrors `IPlayerCatalogStore`/`IMarketStore`).
  `FileDraftStore` writes one `{Id}.json` per draft (body = `GraphSerializer` output) under an
  **injected root dir**, so it stays engine-free; the UI composes it with
  `Application.persistentDataPath/BotDrafts`. Tests use an in-memory fake plus a temp-dir run of
  the real file store.

### Status (epic)

- **Done:** E1 (model, validation, serialization) + E2 (interpreter + `GraphBot`/sensor
  runtime) + **E3.1** (authoring data bridge + draft store), all unit-tested.
- **Not built:** the rest of E3 — the player-facing node editor scene (E3.2), test-run
  hook (E3.3), and Workshop shell + My Creations screen (E3.4). The `Assets/Dev/Bagus`
  prototype (`ModuleUI`/`BoardManager`/`PortUI`) is UI-only and not yet wired to a
  `GraphDocument`.
- **Sharp edge / open thread:** nothing calls `AssignGraphBot` in a real match yet, so the
  battle-drive path (a `GraphBot` actually steering a robot end-to-end) is unexercised
  pending E3's "test run" or a temporary hook.

---

## 6. Services layer

Backend-agnostic seam (see `CLAUDE.md` for the full contract):
- Composition root: `GameServices` (`Assets/Scripts/Services/Core/`). Swap Local* stubs
  for Photon/UGS **only here**.
- Surfaces: `IAuthService`, `IPlayerDataService`, `ICatalogService`, `ILeaderboardService`
  (`ILeaderboardService` lives in `Assets/Scripts/Leaderboard/`).
- `GameManager` boots services + session; keeps scene routing/profiles, not backend wiring.

---

## 7. Proposed play/train split (roadmap)

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
