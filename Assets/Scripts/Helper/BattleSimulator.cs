using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SumoManager;
using SumoBot;
using System.Linq;
using SumoCore;
using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Unity.VisualScripting;
using Newtonsoft.Json;
using UnityEngine.Serialization;

namespace SumoHelper
{
    public enum SimulatorMode
    {
        Simple,
        Advanced
    }

    public class BattleSimulator : MonoBehaviour
    {
        [Header("Simulator Mode")]
        public SimulatorMode Mode = SimulatorMode.Simple;

        [Header("Simple Mode Settings")]
        public int TotalSimulations = 5;
        public float SimpleTimeScale = 1f;
        public int SwapAIInterval = 0;
        public bool QuitAfterDone = true;

        [Header("Advanced Mode Settings")]
        public bool IgnoreResume = false;
        public float DefaultTimeScale = 2f;
        public bool SimulationOnStart = false;

        public int RoundCountdown = 3;
        public SimulationSetting Setting;

        [Header("Pacing Simulation")]
        [Tooltip("When enabled, generates Focus-vs-Rest matchups swept across every Sim Targets x Sim Constraints combination, applying pacing (action filtering) only to the focus bot side. Replaces the default full round-robin matchup generation.")]
        public bool PacingSimulation = false;
        [Tooltip("Bot IDs (from Setting.SelectedAgents) explicitly marked as 'focus' bots, set via the Focus Bot Selection checkboxes in the inspector. Everyone else in SelectedAgents is 'rest'.")]
        [FormerlySerializedAs("TopBotIDs")]
        [HideInInspector] public string[] FocusBotIDs = new string[] { };
        [Tooltip("When enabled, also generates matchups between two Focus bots (in addition to Focus-vs-Rest), each one taking a turn as the pacing-filtered side against the other. Requires at least 2 Focus bots selected.")]
        public bool IncludeFocusMatchups = false;
        [Tooltip("Resources-relative folder of pacing TARGET curves to sweep (only ThreatTargets/TempoTargets are used from these files).")]
        public string SimTargetsFolder = "Pacing/Sim_Targets/60s";
        [Tooltip("Resources-relative folder of pacing CONSTRAINT sets to sweep (only GlobalConstraints is used from these files).")]
        public string SimConstraintsFolder = "Pacing/Sim_Constraints";
        public int PacingSegmentDuration = 1;
        public int PacingCollisionWindow = 3;
        public float PacingMin = 0f;
        public float PacingMax = 0.474f;

        [Header("Bot Exclusion (Performance Optimization)")]
        [Tooltip("Exclude heavy ML bots from simulation. Uncheck to include all bots.")]
        public bool ExcludeHeavyMLBots = true;

        // ML bots that require heavy processing
        // Comment out any bot ID below to re-enable it in simulations
        private readonly string[] HeavyMLBotIDs = new string[]
        {
            "Bot_SLM",   // Bot_SLM_ActionGPT - Small Language Model
            "Bot_LLM",   // Bot_LLM_ActionGPT - Large Language Model
            "Bot_MLP"    // Bot_ML_Classification - Multi-Layer Perceptron
        };

        private List<Bot> Agents = new();
        private List<BattleConfig> _configs;
        private int currentConfigIndex = 0;
        private int firstConfigIndex = 0;
        private SimulationCheckpoint checkpoint;

        #region No-Graphic simulation setting
        private static int ConfigStart = -1;
        private static int ConfigEnd = -1;
        private static int ConfigIndex = -1;
        private static float ConfigTimeScale = -1f;
        private static bool Batched = false;

        // Pacing Simulation overrides (unset unless passed on the command line - batch-mode
        // builds have no Inspector to set these through, since BattleSimulatorEditor only
        // renders in-editor). Applied onto the instance fields in RunAdvancedSimulations(),
        // before the checkpoint is loaded, so resume validity checks see the overridden values.
        private static bool? OverridePacingSimulation = null;
        private static string OverrideSimTargetsFolder = null;
        private static string OverrideSimConstraintsFolder = null;
        private static int? OverridePacingSegmentDuration = null;
        private static int? OverridePacingCollisionWindow = null;
        private static float? OverridePacingMin = null;
        private static float? OverridePacingMax = null;
        private static string[] OverrideFocusBotIDs = null;
        private static bool? OverrideIncludeFocusMatchups = null;
        #endregion

        void OnDisable()
        {
            if (Mode == SimulatorMode.Advanced)
                BattleManager.Instance.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleStateChanged);
        }

        private IEnumerator RunSimpleSimulations()
        {
            // Delay for preparing
            yield return new WaitForSeconds(0.5f);

            for (int i = 0; i < TotalSimulations; i++)
            {
                if (SwapAIInterval > 0 && i > 0 && (i % SwapAIInterval == 0))
                {
                    BattleManager.Instance.BotManager.Swap();
                }

                yield return new WaitForSeconds(1);

                Time.timeScale = SimpleTimeScale;

                if (SimulationOnStart || i > 0)
                {
                    BattleManager.Instance.Battle_Start();
                }

                while (BattleManager.Instance.CurrentState != BattleState.PostBattle_ShowResult)
                {
                    yield return null; // wait frame
                }

                // Time.timeScale = 1;
                yield return new WaitForSeconds(1);
                yield return new WaitForEndOfFrame(); // Delay if needed
            }

            Logger.Info("[Simple Simulation] Complete.", true);
#if UNITY_EDITOR
            if (QuitAfterDone)
                UnityEditor.EditorApplication.isPlaying = false;
            else
                Time.timeScale = 1;
#else
            Application.Quit();
#endif
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        static void ReadArgs()
        {
            string[] args = Environment.GetCommandLineArgs();

            foreach (string arg in args)
            {
                if (arg.StartsWith("--configStart="))
                {
                    string value = arg.Substring("--configStart=".Length);
                    if (int.TryParse(value, out int start))
                        ConfigStart = start;
                }

                if (arg.StartsWith("--configEnd="))
                {
                    string value = arg.Substring("--configEnd=".Length);
                    if (int.TryParse(value, out int end))
                        ConfigEnd = end;
                }

                if (arg.StartsWith("--configIndex="))
                {
                    string value = arg.Substring("--configIndex=".Length);
                    if (int.TryParse(value, out int index))
                    {
                        ConfigIndex = index;
                        ConfigStart = index;
                        ConfigEnd = index + 1;
                    }
                }

                if (arg.StartsWith("--configTimeScale="))
                {
                    string value = arg.Substring("--configTimeScale=".Length);
                    if (float.TryParse(value, out float timeScale))
                        ConfigTimeScale = timeScale;
                }

                if (arg.StartsWith("--batchLogFile="))
                {
                    string value = arg.Substring("--batchLogFile=".Length);
                    if (value == "nul")
                        Logger.BatchLogPath = null;
                    else
                        Logger.BatchLogPath = Path.Combine(Application.persistentDataPath, value);
                }

                if (arg.StartsWith("--pacingSimulation="))
                {
                    string value = arg.Substring("--pacingSimulation=".Length);
                    if (bool.TryParse(value, out bool pacingSimulation))
                        OverridePacingSimulation = pacingSimulation;
                }

                if (arg.StartsWith("--simTargetsFolder="))
                    OverrideSimTargetsFolder = arg.Substring("--simTargetsFolder=".Length);

                if (arg.StartsWith("--simConstraintsFolder="))
                    OverrideSimConstraintsFolder = arg.Substring("--simConstraintsFolder=".Length);

                if (arg.StartsWith("--pacingSegmentDuration="))
                {
                    string value = arg.Substring("--pacingSegmentDuration=".Length);
                    if (int.TryParse(value, out int segmentDuration))
                        OverridePacingSegmentDuration = segmentDuration;
                }

                if (arg.StartsWith("--pacingCollisionWindow="))
                {
                    string value = arg.Substring("--pacingCollisionWindow=".Length);
                    if (int.TryParse(value, out int collisionWindow))
                        OverridePacingCollisionWindow = collisionWindow;
                }

                if (arg.StartsWith("--pacingMin="))
                {
                    string value = arg.Substring("--pacingMin=".Length);
                    if (float.TryParse(value, out float pacingMin))
                        OverridePacingMin = pacingMin;
                }

                if (arg.StartsWith("--pacingMax="))
                {
                    string value = arg.Substring("--pacingMax=".Length);
                    if (float.TryParse(value, out float pacingMax))
                        OverridePacingMax = pacingMax;
                }

                if (arg.StartsWith("--focusBotIDs="))
                {
                    string value = arg.Substring("--focusBotIDs=".Length);
                    OverrideFocusBotIDs = value.Split(',').Select(s => s.Trim()).Where(s => s.Length > 0).ToArray();
                }

                if (arg.StartsWith("--includeFocusMatchups="))
                {
                    string value = arg.Substring("--includeFocusMatchups=".Length);
                    if (bool.TryParse(value, out bool includeFocusMatchups))
                        OverrideIncludeFocusMatchups = includeFocusMatchups;
                }

                if (ConfigStart > -1 && ConfigEnd > -1 && Application.isBatchMode)
                {
                    Batched = true;
                }
            }

            if (ConfigIndex > -1)
            {
                Logger.Info($"[BatchedCommandLineArgs] ConfigIndex={ConfigIndex}, ConfigTimeScale={ConfigTimeScale}", true);
            }
            else
            {
                Logger.Info($"[BatchedCommandLineArgs] ConfigStart={ConfigStart}, ConfigEnd={ConfigEnd}, ConfigTimeScale={ConfigTimeScale}", true);
            }
        }

        public void PrepareSimulation()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = true;
#endif
            BGMManager.Instance.Stop(true);
            BGMManager.Instance.gameObject.SetActive(false);
            SFXManager.Instance.gameObject.SetActive(false);
            VFXManager.Instance.gameObject.SetActive(false);

            if (Mode == SimulatorMode.Simple)
            {
                Application.runInBackground = true;
                Time.timeScale = SimpleTimeScale;
                StartCoroutine(RunSimpleSimulations());
            }
            else
                RunAdvancedSimulations();

        }

        private void RunAdvancedSimulations()
        {
            ApplyPacingCommandLineOverrides();

            checkpoint = LoadCheckpoint(IgnoreResume);

            if (Setting.Timers.Length == 0)
                throw new Exception("Timers can't be empty");
            if (Setting.RoundSystem.Length == 0)
                throw new Exception("RoundSystem can't be empty");
            if (Setting.SelectedAgents.Length <= 1)
                throw new Exception("SelectedAgents must > 1");

            SelectAgents();

            Application.runInBackground = true;
            BattleManager.Instance.CountdownTime = RoundCountdown;
            BattleManager.Instance.BotManager.LeftEnabled = true;
            BattleManager.Instance.BotManager.RightEnabled = true;

            _configs = GenerateConfigs(Agents);

            // Generate config index mapping for easier re-simulation
            if (Mode == SimulatorMode.Advanced)
            {
                GenerateConfigIndexMapping(_configs, Agents);
            }

            if (Batched)
            {
                Logger.Info($"[Simulation] Applied Config: StartAt {ConfigStart}, EndAt {ConfigEnd}", true);
                currentConfigIndex = ConfigStart;
                firstConfigIndex = ConfigStart;
                checkpoint.ConfigIndex = ConfigStart;
                checkpoint.Iteration = 0;
            }
            else
            {
                SaveCheckpoint(checkpoint);
            }

            checkpoint.TotalConfigs = _configs.Count();

            // Validate config index is within bounds
            if (_configs.Count == 0)
            {
                Logger.Error("[Simulation] No configurations generated. Check SimulationSetting and SelectedAgents.");
                return;
            }

            if (currentConfigIndex >= _configs.Count)
            {
                Logger.Error($"[Simulation] ConfigIndex {currentConfigIndex} is out of range. Total configs: {_configs.Count} (0-{_configs.Count - 1})");
                return;
            }

            ApplyConfig(_configs[currentConfigIndex]);

            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleStateChanged);
        }

        /// <summary>
        /// Applies any --pacing*/--simTargetsFolder/--simConstraintsFolder/--focusBotIDs/
        /// --includeFocusMatchups command-line arguments onto the instance's Pacing Simulation
        /// fields, so headless batch-mode builds can override them per-launch instead of being
        /// stuck with whatever was serialized into the scene at build time. Only fields whose
        /// override was actually passed are touched; everything else keeps its Inspector value.
        /// Must run before LoadCheckpoint() so the resume-validity check compares against the
        /// overridden values, not the stale ones baked into the scene.
        /// </summary>
        private void ApplyPacingCommandLineOverrides()
        {
            bool anyOverride = OverridePacingSimulation.HasValue || OverrideSimTargetsFolder != null ||
                OverrideSimConstraintsFolder != null || OverridePacingSegmentDuration.HasValue ||
                OverridePacingCollisionWindow.HasValue || OverridePacingMin.HasValue || OverridePacingMax.HasValue ||
                OverrideFocusBotIDs != null || OverrideIncludeFocusMatchups.HasValue;

            if (!anyOverride)
                return;

            if (OverridePacingSimulation.HasValue) PacingSimulation = OverridePacingSimulation.Value;
            if (OverrideSimTargetsFolder != null) SimTargetsFolder = OverrideSimTargetsFolder;
            if (OverrideSimConstraintsFolder != null) SimConstraintsFolder = OverrideSimConstraintsFolder;
            if (OverridePacingSegmentDuration.HasValue) PacingSegmentDuration = OverridePacingSegmentDuration.Value;
            if (OverridePacingCollisionWindow.HasValue) PacingCollisionWindow = OverridePacingCollisionWindow.Value;
            if (OverridePacingMin.HasValue) PacingMin = OverridePacingMin.Value;
            if (OverridePacingMax.HasValue) PacingMax = OverridePacingMax.Value;
            if (OverrideFocusBotIDs != null) FocusBotIDs = OverrideFocusBotIDs;
            if (OverrideIncludeFocusMatchups.HasValue) IncludeFocusMatchups = OverrideIncludeFocusMatchups.Value;

            Logger.Info($"[BattleSimulator] Applied pacing command-line overrides: PacingSimulation={PacingSimulation}, " +
                $"SimTargetsFolder={SimTargetsFolder}, SimConstraintsFolder={SimConstraintsFolder}, " +
                $"PacingSegmentDuration={PacingSegmentDuration}, PacingCollisionWindow={PacingCollisionWindow}, " +
                $"PacingMin={PacingMin}, PacingMax={PacingMax}, FocusBotIDs=[{string.Join(",", FocusBotIDs ?? new string[0])}], " +
                $"IncludeFocusMatchups={IncludeFocusMatchups}", true);
        }

        private void OnBattleStateChanged(EventParameter param)
        {
            if (param.BattleState == BattleState.PreBatle_Preparing)
            {
                StartCoroutine(RunSimulations());
            }
            else if (param.BattleState == BattleState.Battle_Preparing)
            {
                var cfg = _configs[currentConfigIndex];
                SetBot(cfg);
            }
        }

        private void SelectAgents()
        {
            var botTypes = BotUtility.GetAllBotInstances();
            Logger.Info($"[SelectAgents] BotUtility returned {botTypes.Count} bot instances", true);

            var loadedAgents = botTypes.ConvertAll(t => t.ID).ToList();
            Logger.Info($"[SelectAgents] Bot IDs: {string.Join(", ", loadedAgents)}", true);
            Logger.Info($"[SelectAgents] Setting.SelectedAgents.Length: {Setting.SelectedAgents.Length}", true);

            foreach (var botInstance in botTypes)
            {
                Logger.Info($"[SelectAgents] Processing bot: {botInstance?.name ?? "null"}, ID: {botInstance?.ID ?? "null"}", true);

                if (botInstance != null)
                {
                    // Check if bot is excluded (heavy ML bots)
                    if (ExcludeHeavyMLBots && HeavyMLBotIDs.Contains(botInstance.ID))
                    {
                        Logger.Info($"[SelectAgents] Excluding heavy ML bot '{botInstance.ID}' (ExcludeHeavyMLBots=true)", true);
                        continue;
                    }

                    if (Setting.SelectedAgents.Length > 0)
                    {
                        bool contains = Setting.SelectedAgents.Contains(botInstance.ID);
                        Logger.Info($"[SelectAgents] SelectedAgents contains '{botInstance.ID}': {contains}", true);
                        if (contains)
                            Agents.Add(botInstance);
                    }
                    else
                    {
                        // If no agents selected, add all agents (except excluded ones)
                        Logger.Info($"[SelectAgents] No agents filter, adding {botInstance.ID}", true);
                        Agents.Add(botInstance);
                    }
                }
                else
                {
                    Logger.Warning($"[SelectAgents] Bot instance is null!");
                }
            }
            Logger.Info($"[Simulation] Loaded {Agents.Count} agents", true);
        }

        private IEnumerator RunSimulations()
        {
            yield return new WaitForSecondsRealtime(0.5f);

            for (currentConfigIndex = checkpoint.ConfigIndex; currentConfigIndex < (Batched ? ConfigEnd : _configs.Count); currentConfigIndex++)
            {
                Time.timeScale = 1;

                BattleConfig cfg = _configs[currentConfigIndex];
                var (resumeAt, gameLogs) = GetResumeIterations(cfg);
                checkpoint.Iteration = resumeAt;
                if (resumeAt >= cfg.Iteration)
                {
                    Logger.Info($"[Simulation][Skip] {currentConfigIndex} already completed {cfg.Iteration} iterations.", true);
                    continue;
                }

                if (currentConfigIndex > firstConfigIndex)
                {
                    ApplyConfig(_configs[currentConfigIndex]);
                }

                LogManager.Log.Games = gameLogs;
                yield return new WaitForEndOfFrame();

                // if (!cfg.AgentLeft.UseAsync && !cfg.AgentRight.UseAsync)
                // Apply ConfigTimeScale override if provided via command line, otherwise use cfg.TimeScale
                Time.timeScale = ConfigTimeScale > 0 ? ConfigTimeScale : cfg.TimeScale;

                for (int iter = resumeAt; iter < cfg.Iteration; iter++)
                {
                    Logger.Info($"[Simulation] Config {currentConfigIndex}/{_configs.Count}, Iteration {iter}/{cfg.Iteration} | " +
                              $"{cfg.AgentLeft.ID} vs {cfg.AgentRight.ID} | " +
                              $"Timer={cfg.Timer}, ActInterval={cfg.ActionInterval}, Round={cfg.RoundSystem}, SkillLeft={cfg.SkillSetLeft}, SkillRight={cfg.SkillSetRight}", true);

                    yield return new WaitForSecondsRealtime(1);

                    if (SimulationOnStart || currentConfigIndex > 0 || iter > 0)
                    {
                        BattleManager.Instance.Battle_Start();
                    }

                    while (BattleManager.Instance.CurrentState != BattleState.PostBattle_ShowResult)
                    {
                        yield return new WaitForEndOfFrame();
                    }

                    yield return new WaitForSecondsRealtime(1);
                    checkpoint.Iteration = iter;
                    checkpoint.ConfigIndex = currentConfigIndex;
                    if (!Batched)
                        SaveCheckpoint(checkpoint);
                    yield return new WaitForEndOfFrame();

                }
                Logger.Info($"[Simulation] Config {currentConfigIndex}/{_configs.Count}, Completed | " +
                              $"{cfg.AgentLeft.ID} vs {cfg.AgentRight.ID} | " +
                              $"Timer={cfg.Timer}, ActInterval={cfg.ActionInterval}, Round={cfg.RoundSystem}, SkillLeft={cfg.SkillSetLeft}, SkillRight={cfg.SkillSetRight}", true);

                checkpoint.Iteration = 0;
            }

            Logger.Info("[Simulation] All simulations complete.", true);

#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }

        private void ApplyConfig(BattleConfig cfg)
        {
            RoundSystem rs = cfg.RoundSystem;
            BattleManager.Instance.RoundSystem = rs;

            BattleManager.Instance.BattleTime = cfg.Timer;
            BattleManager.Instance.ActionInterval = cfg.ActionInterval;

            ApplyPacingSimulationConfig(cfg);

            var folder = GetFolderStructure(cfg);

            LogManager.UnregisterAction();
            LogManager.InitLog(true, folder);
            LogManager.InitBattle(cfg);

            var newBattle = new Battle(Guid.NewGuid().ToString(), cfg.RoundSystem)
            {
                // Apply previous players to new battle
                LeftPlayer = BattleManager.Instance.Battle.LeftPlayer,
                RightPlayer = BattleManager.Instance.Battle.RightPlayer
            };
            BattleManager.Instance.Battle = newBattle;
        }

        /// <summary>
        /// Pushes this match's pacing setup onto PacingManager before Battle_Preparing fires
        /// (InputManager.InitializeInput reads these fields when it constructs each side's
        /// PacingHandler). Both sides load the same swept target/constraint files - only the
        /// focus bot's side gets action filtering and NN candidates enabled, the other side loads
        /// the same PacingTarget purely for measurement/logging with filtering off. This avoids
        /// depending on PacingManager's normal fallback config (Left/RightFileName from
        /// Resources/Pacing/Constraints), which isn't guaranteed to exist during pacing sweeps.
        /// Clears the overrides when Pacing Simulation is disabled so stale values from a
        /// previous run can't leak into a manual/non-sim session.
        /// </summary>
        private void ApplyPacingSimulationConfig(BattleConfig cfg)
        {
            // Use BattleManager's own reference instead of PacingManager.Instance: this method
            // runs synchronously inside BattleManager.OnEnable() (via PrepareSimulation()) on the
            // very first battle, and PacingManager sits after BattleManager in that GameObject's
            // component list - its own Awake() (which sets Instance) hasn't necessarily run yet
            // at this point, even though GetComponent<PacingManager>() (which BattleManager.OnEnable
            // already called into its own PacingManager field) works regardless of Awake timing.
            var pacingManager = BattleManager.Instance.PacingManager;
            if (pacingManager == null)
            {
                Logger.Warning("[BattleSimulator] PacingManager component not found on BattleManager's GameObject - pacing simulation config not applied.");
                return;
            }

            if (!PacingSimulation)
            {
                pacingManager.LeftSimTargetPath = null;
                pacingManager.LeftSimConstraintPath = null;
                pacingManager.RightSimTargetPath = null;
                pacingManager.RightSimConstraintPath = null;
                return;
            }

            pacingManager.MinPacing = PacingMin;
            pacingManager.MaxPacing = PacingMax;

            string targetPath = $"{SimTargetsFolder}/{cfg.PacingTargetFileName}";
            string constraintPath = $"{SimConstraintsFolder}/{cfg.PacingConstraintFileName}";
            bool focusIsLeft = cfg.PacingSide == "Left";

            Logger.Info($"[BattleSimulator] Applying pacing sim config: target={targetPath}, constraint={constraintPath}");

            pacingManager.LeftSimTargetPath = targetPath;
            pacingManager.LeftSimConstraintPath = constraintPath;
            pacingManager.LeftActionFiltering = focusIsLeft;
            pacingManager.LeftNNCandidates = focusIsLeft;
            pacingManager.LeftMCTSCandidates = false;
            pacingManager.LeftSegmentDuration = cfg.PacingSegmentDuration;
            pacingManager.LeftCollisionWindowDuration = cfg.PacingCollisionWindow;

            pacingManager.RightSimTargetPath = targetPath;
            pacingManager.RightSimConstraintPath = constraintPath;
            pacingManager.RightActionFiltering = !focusIsLeft;
            pacingManager.RightNNCandidates = !focusIsLeft;
            pacingManager.RightMCTSCandidates = false;
            pacingManager.RightSegmentDuration = cfg.PacingSegmentDuration;
            pacingManager.RightCollisionWindowDuration = cfg.PacingCollisionWindow;
        }

        private void SetBot(BattleConfig cfg)
        {
            BattleManager.Instance.BotManager.Assign(cfg.AgentLeft, PlayerSide.Left, cfg.SkillSetLeft, false);
            BattleManager.Instance.BotManager.Assign(cfg.AgentRight, PlayerSide.Right, cfg.SkillSetRight, false);
        }

        private List<BattleConfig> GenerateConfigs(List<Bot> agents)
        {
            if (PacingSimulation)
                return GeneratePacingSimulationConfigs(agents);

            var configs = new List<BattleConfig>();

            for (int i = 0; i < agents.Count; i++)
            {
                for (int j = 0; j < agents.Count; j++)
                {
                    if (i == j) continue; // skip self-matchups

                    foreach (var roundSystem in Setting.RoundSystem)
                    {
                        foreach (var timer in Setting.Timers)
                        {
                            foreach (var interval in Setting.ActionIntervals)
                            {
                                if (Setting.Skills.Length > 0)
                                {
                                    for (int leftSkillI = 0; leftSkillI < Setting.Skills.Length; leftSkillI++)
                                    {
                                        for (int rightSkillI = 0; rightSkillI < Setting.Skills.Length; rightSkillI++)
                                        {
                                            configs.Add(new BattleConfig
                                            {
                                                AgentLeft = agents[i],
                                                AgentRight = agents[j],
                                                Timer = timer,
                                                ActionInterval = interval,
                                                SkillSetLeft = Setting.Skills[leftSkillI],
                                                SkillSetRight = Setting.Skills[rightSkillI],
                                                Iteration = Setting.Iteration,
                                                TimeScale = DefaultTimeScale,
                                                RoundSystem = roundSystem
                                            });
                                        }
                                    }
                                }
                                else
                                {
                                    configs.Add(new BattleConfig
                                    {
                                        AgentLeft = agents[i],
                                        AgentRight = agents[j],
                                        Timer = timer,
                                        ActionInterval = interval,
                                        Iteration = Setting.Iteration,
                                        TimeScale = DefaultTimeScale,
                                        RoundSystem = roundSystem
                                    });
                                }

                            }
                        }
                    }
                }
            }

            Logger.Info($"Generated configs: {configs.Count}", true);
            Logger.Info($"Game will run {configs.Aggregate(0, (sum, cfg) => sum + cfg.Iteration)} matches in total.", true);
            return configs;
        }

        /// <summary>
        /// Generates Focus-vs-Rest matchups (both sides mirrored) swept across every
        /// Sim_Targets x Sim_Constraints combination. "Focus" bots are whichever of
        /// Setting.SelectedAgents are also listed in FocusBotIDs (set via the Focus Bot
        /// Selection checkboxes); "Rest" is everyone else in SelectedAgents. When
        /// IncludeFocusMatchups is also enabled, Focus bots are additionally swept against
        /// each other (Focus-vs-Focus), each one taking a turn as the pacing side.
        /// Only the focused bot's side is marked (via BattleConfig.PacingSide) to receive
        /// pacing; ApplyPacingSimulationConfig() applies that to PacingManager before each
        /// match starts.
        /// </summary>
        private List<BattleConfig> GeneratePacingSimulationConfigs(List<Bot> agents)
        {
            var configs = new List<BattleConfig>();

            var focusSet = new HashSet<string>(FocusBotIDs ?? new string[0]);
            var focusAgents = agents.Where(a => focusSet.Contains(a.ID)).ToList();
            var restAgents = agents.Where(a => !focusSet.Contains(a.ID)).ToList();

            if (focusAgents.Count == 0 || restAgents.Count == 0)
            {
                Logger.Error($"[Simulation][PacingSimulation] Requires at least 1 focus agent and 1 rest agent (Focus={focusAgents.Count}, Rest={restAgents.Count}). Check Focus Bot Selection in the inspector.");
                return configs;
            }

            if (IncludeFocusMatchups && focusAgents.Count < 2)
            {
                Logger.Warning($"[Simulation][PacingSimulation] Include Focus Matchups is enabled but only {focusAgents.Count} focus bot(s) selected - need at least 2 to generate Focus-vs-Focus matchups.");
            }

            var targetAssets = Resources.LoadAll<TextAsset>(SimTargetsFolder).OrderBy(a => a.name).ToList();
            var constraintAssets = Resources.LoadAll<TextAsset>(SimConstraintsFolder).OrderBy(a => a.name).ToList();

            if (targetAssets.Count == 0)
            {
                Logger.Error($"[Simulation][PacingSimulation] No pacing target files found under Resources/{SimTargetsFolder}.");
                return configs;
            }
            if (constraintAssets.Count == 0)
            {
                Logger.Error($"[Simulation][PacingSimulation] No pacing constraint files found under Resources/{SimConstraintsFolder}.");
                return configs;
            }

            // Focus vs Rest: every focus bot swept against every non-focus bot, both directions.
            foreach (var focusBot in focusAgents)
                foreach (var restBot in restAgents)
                    AddMatchupConfigs(configs, focusBot, restBot, targetAssets, constraintAssets);

            // Focus vs Focus (optional): sweep focus bots against each other too. Iterating every
            // ordered pair (A,B) and (B,A) - each contributing "A focused" and "B focused" configs
            // respectively - covers both bots as the focused side without generating duplicates.
            if (IncludeFocusMatchups)
            {
                foreach (var focusBotA in focusAgents)
                    foreach (var focusBotB in focusAgents)
                    {
                        if (focusBotA == focusBotB) continue;
                        AddMatchupConfigs(configs, focusBotA, focusBotB, targetAssets, constraintAssets);
                    }
            }

            Logger.Info($"[Simulation][PacingSimulation] Generated configs: {configs.Count} (Focus={focusAgents.Count}, Rest={restAgents.Count}, IncludeFocusMatchups={IncludeFocusMatchups}, Targets={targetAssets.Count}, Constraints={constraintAssets.Count})", true);
            Logger.Info($"Game will run {configs.Aggregate(0, (sum, cfg) => sum + cfg.Iteration)} matches in total.", true);
            return configs;
        }

        /// <summary>
        /// Generates both directional configs for a pacing-focused bot against an opponent -
        /// focusBot as Left (focused) vs opponent as Right, and focusBot as Right (focused) vs
        /// opponent as Left - swept across every round system / timer / action interval / target
        /// / constraint combination. Shared by the Focus-vs-Rest and Focus-vs-Focus sweeps.
        /// </summary>
        private void AddMatchupConfigs(List<BattleConfig> configs, Bot focusBot, Bot opponentBot, List<TextAsset> targetAssets, List<TextAsset> constraintAssets)
        {
            var pairings = new (Bot left, Bot right, PlayerSide focusSide)[]
            {
                (focusBot, opponentBot, PlayerSide.Left),
                (opponentBot, focusBot, PlayerSide.Right),
            };

            foreach (var (leftBot, rightBot, focusSide) in pairings)
            {
                foreach (var roundSystem in Setting.RoundSystem)
                {
                    foreach (var timer in Setting.Timers)
                    {
                        foreach (var interval in Setting.ActionIntervals)
                        {
                            foreach (var targetAsset in targetAssets)
                            {
                                foreach (var constraintAsset in constraintAssets)
                                {
                                    AddPacingSimulationConfigs(configs, leftBot, rightBot, focusSide, roundSystem, timer, interval, targetAsset.name, constraintAsset.name);
                                }
                            }
                        }
                    }
                }
            }
        }

        private void AddPacingSimulationConfigs(List<BattleConfig> configs, Bot leftBot, Bot rightBot, PlayerSide focusSide, RoundSystem roundSystem, int timer, float interval, string targetFileName, string constraintFileName)
        {
            if (Setting.Skills.Length > 0)
            {
                for (int leftSkillI = 0; leftSkillI < Setting.Skills.Length; leftSkillI++)
                {
                    for (int rightSkillI = 0; rightSkillI < Setting.Skills.Length; rightSkillI++)
                    {
                        configs.Add(new BattleConfig
                        {
                            AgentLeft = leftBot,
                            AgentRight = rightBot,
                            Timer = timer,
                            ActionInterval = interval,
                            SkillSetLeft = Setting.Skills[leftSkillI],
                            SkillSetRight = Setting.Skills[rightSkillI],
                            Iteration = Setting.Iteration,
                            TimeScale = DefaultTimeScale,
                            RoundSystem = roundSystem,
                            PacingTargetFileName = targetFileName,
                            PacingConstraintFileName = constraintFileName,
                            PacingSide = focusSide.ToString(),
                            PacingSegmentDuration = PacingSegmentDuration,
                            PacingCollisionWindow = PacingCollisionWindow,
                            PacingMax = PacingMax,
                            PacingMin = PacingMin
                        });
                    }
                }
            }
            else
            {
                configs.Add(new BattleConfig
                {
                    AgentLeft = leftBot,
                    AgentRight = rightBot,
                    Timer = timer,
                    ActionInterval = interval,
                    Iteration = Setting.Iteration,
                    TimeScale = DefaultTimeScale,
                    RoundSystem = roundSystem,
                    PacingTargetFileName = targetFileName,
                    PacingConstraintFileName = constraintFileName,
                    PacingSide = focusSide.ToString(),
                    PacingSegmentDuration = PacingSegmentDuration,
                    PacingCollisionWindow = PacingCollisionWindow,
                    PacingMax = PacingMax,
                    PacingMin = PacingMin
                });
            }
        }

        private void GenerateConfigIndexMapping(List<BattleConfig> configs, List<Bot> agents)
        {
            try
            {
                // Create mapping file path inside the checkpoint's batch folder
                string logsPath = Path.Combine(Application.persistentDataPath, "Logs", "Batch", checkpoint.ID);
                if (!Directory.Exists(logsPath))
                {
                    Directory.CreateDirectory(logsPath);
                }

                string mappingFilePath = Path.Combine(logsPath, "config_index_mapping.txt");

                using (StreamWriter writer = new StreamWriter(mappingFilePath))
                {
                    writer.WriteLine("=============================================================");
                    writer.WriteLine($"Config Index Mapping - Simulation ID: {checkpoint.ID}");
                    writer.WriteLine($"Generated: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                    writer.WriteLine($"Total Configs: {configs.Count}");
                    writer.WriteLine($"Total Agents: {agents.Count}");
                    writer.WriteLine("=============================================================");
                    writer.WriteLine();

                    // Group configs by bot (AgentLeft)
                    var botGroups = new Dictionary<string, List<(int index, BattleConfig config)>>();

                    for (int i = 0; i < configs.Count; i++)
                    {
                        var cfg = configs[i];
                        string botId = cfg.AgentLeft.ID;

                        if (!botGroups.ContainsKey(botId))
                        {
                            botGroups[botId] = new List<(int, BattleConfig)>();
                        }

                        botGroups[botId].Add((i, cfg));
                    }

                    // Write bot sections
                    foreach (var bot in agents)
                    {
                        if (!botGroups.ContainsKey(bot.ID))
                            continue;

                        var botConfigs = botGroups[bot.ID];
                        int startIndex = botConfigs[0].index;
                        int endIndex = botConfigs[botConfigs.Count - 1].index;

                        writer.WriteLine($"{bot.ID} (StartIndex: {startIndex}, EndIndex: {endIndex}, Total: {botConfigs.Count})");
                        writer.WriteLine(new string('-', 80));

                        foreach (var (index, config) in botConfigs)
                        {
                            string configName = $"Timer_{config.Timer}__ActInterval_{config.ActionInterval}__Round_{config.RoundSystem}__SkillLeft_{config.SkillSetLeft}__SkillRight_{config.SkillSetRight}";
                            writer.WriteLine($"  [{index:D5}] {config.AgentLeft.ID}_vs_{config.AgentRight.ID} | {configName}");
                        }

                        writer.WriteLine();
                    }

                    writer.WriteLine("=============================================================");
                    writer.WriteLine("SUMMARY BY BOT");
                    writer.WriteLine("=============================================================");

                    foreach (var bot in agents)
                    {
                        if (!botGroups.ContainsKey(bot.ID))
                            continue;

                        var botConfigs = botGroups[bot.ID];
                        int startIndex = botConfigs[0].index;
                        int endIndex = botConfigs[botConfigs.Count - 1].index;

                        writer.WriteLine($"{bot.ID,-30} StartIndex: {startIndex,5} | EndIndex: {endIndex,5} | Total: {botConfigs.Count,5}");
                    }
                }

                Logger.Info($"[Simulation] Config index mapping saved to: {mappingFilePath}", true);
            }
            catch (Exception ex)
            {
                Logger.Error($"[Simulation] Failed to generate config index mapping: {ex.Message}");
            }
        }



        /// <summary>
        /// Snapshots this simulator's current Pacing Simulation fields onto the checkpoint, so a
        /// later resume attempt can detect whether they've changed since the checkpoint was written.
        /// </summary>
        private void CapturePacingConfig(SimulationCheckpoint checkpoint)
        {
            checkpoint.PacingSimulation = PacingSimulation;
            checkpoint.FocusBotIDs = FocusBotIDs;
            checkpoint.IncludeFocusMatchups = IncludeFocusMatchups;
            checkpoint.SimTargetsFolder = SimTargetsFolder;
            checkpoint.SimConstraintsFolder = SimConstraintsFolder;
            checkpoint.PacingSegmentDuration = PacingSegmentDuration;
            checkpoint.PacingCollisionWindow = PacingCollisionWindow;
            checkpoint.PacingMin = PacingMin;
            checkpoint.PacingMax = PacingMax;
        }

        /// <summary>
        /// Compares this simulator's current Pacing Simulation fields against the checkpoint's
        /// snapshot of them (see CapturePacingConfig). GeneratePacingSimulationConfigs()/
        /// AddMatchupConfigs() derive _configs' count/order directly from these fields, so any
        /// mismatch means resuming from checkpoint.ConfigIndex would silently run the wrong config.
        /// </summary>
        private bool IsPacingConfigEqual(SimulationCheckpoint other)
        {
            if (other == null) return false;
            if (PacingSimulation != other.PacingSimulation) return false;
            if (IncludeFocusMatchups != other.IncludeFocusMatchups) return false;
            if (SimTargetsFolder != other.SimTargetsFolder) return false;
            if (SimConstraintsFolder != other.SimConstraintsFolder) return false;
            if (PacingSegmentDuration != other.PacingSegmentDuration) return false;
            if (PacingCollisionWindow != other.PacingCollisionWindow) return false;
            if (!Mathf.Approximately(PacingMin, other.PacingMin)) return false;
            if (!Mathf.Approximately(PacingMax, other.PacingMax)) return false;

            var mine = new HashSet<string>(FocusBotIDs ?? new string[0]);
            var theirs = new HashSet<string>(other.FocusBotIDs ?? new string[0]);
            if (!mine.SetEquals(theirs)) return false;

            return true;
        }

        private void SaveCheckpoint(SimulationCheckpoint checkpoint)
        {
            string folder = Path.Combine(Application.persistentDataPath, "Settings");
            string path = $"{folder}/simulation_checkpoint.json";
            string json = JsonUtility.ToJson(checkpoint, true);
            File.WriteAllText(path, json);
        }

        private SimulationCheckpoint LoadCheckpoint(bool forceCreate = false)
        {
            string folder = Path.Combine(Application.persistentDataPath, "Settings");
            Directory.CreateDirectory(folder);

            string path = $"{folder}/simulation_checkpoint.json";

            SimulationCheckpoint checkpoint = null;

            if (File.Exists(path))
            {
                string json = File.ReadAllText(path);
                try
                {
                    checkpoint = JsonUtility.FromJson<SimulationCheckpoint>(json);
                }
                catch (Exception)
                {
                    Logger.Info($"[Checkpoint] Failed to read ${path}. Will create a new one", true);
                    checkpoint = null;
                }
            }

            if (checkpoint == null)
            {
                Logger.Info("[Checkpoint] No existing checkpoint found. Creating new checkpoint.", true);
                string newID = DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_batch";
                string createdAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

                checkpoint = new SimulationCheckpoint
                {
                    ID = newID,
                    CreatedAt = createdAt,
                    Setting = Setting,
                    Iteration = 0,
                    ConfigIndex = 0,
                };
                CapturePacingConfig(checkpoint);

                Logger.Info($"[Checkpoint] Created checkpoint ID: {checkpoint.ID} at {checkpoint.CreatedAt}", true);
            }
            else
            {
                // Check if the configuration has changed. Pacing fields are compared too since
                // GeneratePacingSimulationConfigs() derives _configs' count/order directly from
                // them - a mismatch there means checkpoint.ConfigIndex would silently point into
                // a differently-shaped config list on resume.
                bool configurationChanged = !Setting.IsConfigurationEqual(checkpoint.Setting) || !IsPacingConfigEqual(checkpoint);

                if (forceCreate || configurationChanged)
                {
                    if (configurationChanged == true && forceCreate)
                        Logger.Info("[Checkpoint] Found resumable batch simulation but forceCreate=true, creating a new batch.", true);
                    else
                    {
                        Logger.Info("[Checkpoint] Configuration has changed. Resetting checkpoint.", true);
                        Logger.Info($"[Checkpoint] Old Config: Agents={checkpoint.Setting.SelectedAgents?.Length ?? 0}, " +
                                  $"Timers={checkpoint.Setting.Timers?.Length ?? 0}, " +
                                  $"Intervals={checkpoint.Setting.ActionIntervals?.Length ?? 0}, " +
                                  $"Rounds={checkpoint.Setting.RoundSystem?.Length ?? 0}, " +
                                  $"Skills={checkpoint.Setting.Skills?.Length ?? 0}, " +
                                  $"Iteration={checkpoint.Setting.Iteration}, " +
                                  $"PacingSimulation={checkpoint.PacingSimulation}, SimTargetsFolder={checkpoint.SimTargetsFolder}, " +
                                  $"SimConstraintsFolder={checkpoint.SimConstraintsFolder}, FocusBotIDs=[{string.Join(",", checkpoint.FocusBotIDs ?? new string[0])}], " +
                                  $"IncludeFocusMatchups={checkpoint.IncludeFocusMatchups}, PacingMin={checkpoint.PacingMin}, PacingMax={checkpoint.PacingMax}", true);
                        Logger.Info($"[Checkpoint] New Config: Agents={Setting.SelectedAgents?.Length ?? 0}, " +
                                  $"Timers={Setting.Timers?.Length ?? 0}, " +
                                  $"Intervals={Setting.ActionIntervals?.Length ?? 0}, " +
                                  $"Rounds={Setting.RoundSystem?.Length ?? 0}, " +
                                  $"Skills={Setting.Skills?.Length ?? 0}, " +
                                  $"Iteration={Setting.Iteration}, " +
                                  $"PacingSimulation={PacingSimulation}, SimTargetsFolder={SimTargetsFolder}, " +
                                  $"SimConstraintsFolder={SimConstraintsFolder}, FocusBotIDs=[{string.Join(",", FocusBotIDs ?? new string[0])}], " +
                                  $"IncludeFocusMatchups={IncludeFocusMatchups}, PacingMin={PacingMin}, PacingMax={PacingMax}", true);
                    }

                    // Generate new ID and timestamp for reset checkpoint
                    string newID = DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_batch";
                    string createdAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

                    // Reset checkpoint with new settings
                    checkpoint = new SimulationCheckpoint
                    {
                        ID = newID,
                        CreatedAt = createdAt,
                        Setting = Setting,
                        Iteration = 0,
                        ConfigIndex = 0,
                    };
                    CapturePacingConfig(checkpoint);

                    Logger.Info($"[Checkpoint] Created new checkpoint ID: {checkpoint.ID} at {checkpoint.CreatedAt}", true);
                }
                else
                {
                    Logger.Info("[Checkpoint] Configuration matches. Resuming from checkpoint.", true);
                    Logger.Info($"[Checkpoint] Resuming checkpoint ID: {checkpoint.ID}, created at {checkpoint.CreatedAt}", true);
                    Logger.Info($"[Checkpoint] Resuming from Config {checkpoint.ConfigIndex}, Iteration {checkpoint.Iteration}", true);

                    Setting = checkpoint.Setting;
                    currentConfigIndex = checkpoint.ConfigIndex;
                    firstConfigIndex = checkpoint.ConfigIndex;
                }
            }

            return checkpoint;
        }

        private (int, List<LogManager.GameLog>) GetResumeIterations(BattleConfig cfg)
        {
            var path = GetFolderStructure(cfg).ToList();
            path.Insert(0, "Batch");
            path.Insert(0, "Logs");
            path.Insert(0, Application.persistentDataPath);
            string folder = Path.Combine(path.ToArray());
            if (!Directory.Exists(folder))
                return (0, new());

            var files = Directory.GetFiles(folder, "game_*.json");

            List<LogManager.GameLog> gameLogs = new();

            bool isExceed = false;

            for (int i = 0; i < files.Length; i++)
            {
                var file = files[i];

                if (i > cfg.Iteration - 1)
                {
                    Logger.Info($"[BattleSimulator] Config: {currentConfigIndex}, Matchup: {string.Join("/", path)} has exceeding games, 1 index file deleted", true);
                    File.Delete(file);
                    isExceed = true;
                    continue;
                }

                try
                {
                    string json = File.ReadAllText(file);
                    var log = JsonConvert.DeserializeObject<LogManager.GameLog>(json);
                    if (log.Index > -1)
                        if (log.Rounds.Count() >= GetWinningRound(cfg))
                            gameLogs.Add(log);
                }
                catch (Exception e)
                {
                    Logger.Error($"[BattleSimulator] Config: {currentConfigIndex}, Iteration: {gameLogs.Count + 1}. Matchup: {string.Join("/", path)}  Error cause: {e}", true);
                    break;
                }
            }

            // if we already hit target, skip
            if (gameLogs.Count >= cfg.Iteration)
            {
                if (isExceed)
                {
                    return (cfg.Iteration - 1, gameLogs.Take(cfg.Iteration - 1).ToList());
                }
                return (cfg.Iteration, gameLogs);
            }

            // [0, 1, 2, 3, 4, 5] -> Existing logs Count 6
            // Max = Count - 1 => repeat on this iteration 5
            // Max - 1 => load game logs 4
            // Resume at last file (to re-run it)

            var max = gameLogs.Count;
            return (Math.Max(0, max), gameLogs.Take(Math.Max(0, max)).ToList());
        }

        private string[] GetFolderStructure(BattleConfig cfg)
        {
            string configFolder = $"Timer_{cfg.Timer}__ActInterval_{cfg.ActionInterval}__Round_{cfg.RoundSystem}__SkillLeft_{cfg.SkillSetLeft}__SkillRight_{cfg.SkillSetRight}";

            if (PacingSimulation)
                configFolder += $"__Pacing_{cfg.PacingTargetFileName}_constraint_{cfg.PacingConstraintFileName}";

            return new string[]{
                checkpoint.ID,
                $"{cfg.AgentLeft.ID}_vs_{cfg.AgentRight.ID}",
                configFolder,
            };
        }

        private int GetWinningRound(BattleConfig cfg)
        {
            return Enum.GetValues(typeof(RoundSystem)).Cast<RoundSystem>().ToList().IndexOf(cfg.RoundSystem) + 1;

        }
    }

    public class BattleConfig
    {
        public Bot AgentLeft;
        public Bot AgentRight;
        public RoundSystem RoundSystem;
        public int Timer;
        public float ActionInterval;
        public SkillType SkillSetLeft;
        public SkillType SkillSetRight;
        public int LeftSide;
        public int Iteration;
        public float TimeScale;

        // Pacing Simulation only (null/empty when PacingSimulation is disabled).
        // File names only (no folder prefix) - used for both Resources.Load path
        // construction and log folder naming.
        public string PacingTargetFileName;
        public string PacingConstraintFileName;
        public int PacingSegmentDuration;
        public int PacingCollisionWindow;
        public string PacingSide;
        public float PacingMin;
        public float PacingMax;
    }

    [Serializable]
    public class SimulationSetting
    {
        public int Iteration;

        [HideInInspector]
        public int[] Timers;

        [HideInInspector]
        public float[] ActionIntervals;

        [HideInInspector]
        public RoundSystem[] RoundSystem;

        [HideInInspector]
        public SkillType[] Skills;

        [HideInInspector]
        public string[] SelectedAgents = new string[] { };

        public bool IsConfigurationEqual(SimulationSetting other)
        {
            if (other == null) return false;

            // Compare Iteration
            if (Iteration != other.Iteration) return false;

            // Compare Timers
            if (Timers == null && other.Timers != null) return false;
            if (Timers != null && other.Timers == null) return false;
            if (Timers != null && other.Timers != null)
            {
                if (Timers.Length != other.Timers.Length) return false;
                for (int i = 0; i < Timers.Length; i++)
                {
                    if (!other.Timers.Contains(Timers[i])) return false;
                }
            }

            // Compare ActionIntervals
            if (ActionIntervals == null && other.ActionIntervals != null) return false;
            if (ActionIntervals != null && other.ActionIntervals == null) return false;
            if (ActionIntervals != null && other.ActionIntervals != null)
            {
                if (ActionIntervals.Length != other.ActionIntervals.Length) return false;
                for (int i = 0; i < ActionIntervals.Length; i++)
                {
                    if (!other.ActionIntervals.Contains(ActionIntervals[i])) return false;
                }
            }

            // Compare RoundSystem
            if (RoundSystem == null && other.RoundSystem != null) return false;
            if (RoundSystem != null && other.RoundSystem == null) return false;
            if (RoundSystem != null && other.RoundSystem != null)
            {
                if (RoundSystem.Length != other.RoundSystem.Length) return false;
                for (int i = 0; i < RoundSystem.Length; i++)
                {
                    if (!other.RoundSystem.Contains(RoundSystem[i])) return false;
                }
            }

            // Compare Skills
            if (Skills == null && other.Skills != null) return false;
            if (Skills != null && other.Skills == null) return false;
            if (Skills != null && other.Skills != null)
            {
                if (Skills.Length != other.Skills.Length) return false;
                for (int i = 0; i < Skills.Length; i++)
                {
                    if (!other.Skills.Contains(Skills[i])) return false;
                }
            }

            // Compare SelectedAgents
            if (SelectedAgents == null && other.SelectedAgents != null) return false;
            if (SelectedAgents != null && other.SelectedAgents == null) return false;
            if (SelectedAgents != null && other.SelectedAgents != null)
            {
                if (SelectedAgents.Length != other.SelectedAgents.Length) return false;
                for (int i = 0; i < SelectedAgents.Length; i++)
                {
                    if (!other.SelectedAgents.Contains(SelectedAgents[i])) return false;
                }
            }

            return true;
        }
    }

    [Serializable]
    public class SimulationCheckpoint
    {
        public string ID;
        public string CreatedAt;
        public SimulationSetting Setting;
        public int TotalConfigs;
        public int ConfigIndex;
        public int Iteration;

        // Pacing Simulation snapshot, used by IsPacingConfigEqual() to detect config changes
        // that should invalidate a resume (see CapturePacingConfig).
        public bool PacingSimulation;
        public string[] FocusBotIDs;
        public bool IncludeFocusMatchups;
        public string SimTargetsFolder;
        public string SimConstraintsFolder;
        public int PacingSegmentDuration;
        public int PacingCollisionWindow;
        public float PacingMin;
        public float PacingMax;
    }
}

