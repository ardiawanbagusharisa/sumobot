using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SumoManager;
using SumoBot;
using System.Linq;
using SumoCore;
using System;
using System.IO;
using Unity.VisualScripting;
using Newtonsoft.Json;

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

        [Header("Advanced Mode Settings")]
        public float DefaultTimeScale = 2f;
        public bool SimulationOnStart = false;

        public int RoundCountdown = 3;
        public SimulationSetting Setting;

        private List<Bot> Agents = new();
        private List<BattleConfig> _configs;
        private int currentConfigIndex = 0;
        private int firstConfigIndex = 0;
        private SimulationCheckpoint checkpoint;

        #region No-Graphic simulation setting
        private static int ConfigStart = -1;
        private static int ConfigEnd = -1;
        private static bool Batched = false;
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

                if (SimulationOnStart || i > 0)
                {
                    BattleManager.Instance.Battle_Start();
                }

                while (BattleManager.Instance.CurrentState != BattleState.PostBattle_ShowResult)
                {
                    yield return null; // wait frame
                }

                yield return new WaitForSeconds(1);
                yield return new WaitForEndOfFrame(); // Delay if needed
            }

            Logger.Info("[Simple Simulation] Complete.", true);
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
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

                if (arg.StartsWith("--batchLogFile="))
                {
                    string value = arg.Substring("--batchLogFile=".Length);
                    if (value == "nul")
                        Logger.BatchLogPath = null;
                    else
                        Logger.BatchLogPath = Path.Combine(Application.persistentDataPath, value);
                }

                if (ConfigStart > -1 && ConfigEnd > -1 && Application.isBatchMode)
                {
                    Batched = true;
                }
            }

            Logger.Info($"[BatchedCommandLineArgs] ConfigStart={ConfigStart}, ConfigEnd={ConfigEnd}", true);
        }

        public void PrepareSimulation()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = true;
#endif
            BGMManager.Instance.Stop(true);
            SFXManager.Instance.gameObject.SetActive(false);
            BGMManager.Instance.gameObject.SetActive(false);

            if (Mode == SimulatorMode.Simple)
            {
                Time.timeScale = SimpleTimeScale;
                Application.runInBackground = true;
                StartCoroutine(RunSimpleSimulations());
            }
            else
                RunAdvancedSimulations();

        }

        private void RunAdvancedSimulations()
        {
            checkpoint = LoadCheckpoint();

            if (Setting.Timers.Length == 0)
                throw new Exception("Timers can't be empty");
            if (Setting.RoundSystem.Length == 0)
                throw new Exception("RoundSystem can't be empty");
            if (Setting.SelectedAgents.Length == 1)
                throw new Exception("SelectedAgents must > 1");

            SelectAgents();

            Application.runInBackground = true;
            BattleManager.Instance.CountdownTime = RoundCountdown;
            BattleManager.Instance.BotManager.LeftEnabled = true;
            BattleManager.Instance.BotManager.RightEnabled = true;

            _configs = GenerateConfigs(Agents);

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

            ApplyConfig(_configs[currentConfigIndex]);

            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleStateChanged);
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
            var botTypes = BotUtility.GetAllBotTypes();
            var loadedAgents = botTypes.ConvertAll(t => t.Name).ToList();

            foreach (var item in botTypes)
            {
                var botInstance = ScriptableObject.CreateInstance(item) as Bot;
                if (botInstance != null)
                {
                    if (Setting.SelectedAgents.Length > 0)
                    {
                        if (Setting.SelectedAgents.Contains(botInstance.ID))
                            Agents.Add(botInstance);
                    }
                    else
                    {
                        // If no agents selected, add all agents
                        Agents.Add(botInstance);
                    }
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
                Time.timeScale = cfg.TimeScale;

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

            var folder = GetFolderStructure(cfg);

            LogManager.UnregisterAction();
            LogManager.InitLog(true, folder);
            LogManager.InitBattle(cfg);

            var newBattle = new Battle(Guid.NewGuid().ToString(), cfg.RoundSystem);

            // Apply previous players to new battle
            newBattle.LeftPlayer = BattleManager.Instance.Battle.LeftPlayer;
            newBattle.RightPlayer = BattleManager.Instance.Battle.RightPlayer;
            BattleManager.Instance.Battle = newBattle;
        }

        private void SetBot(BattleConfig cfg)
        {
            BattleManager.Instance.BotManager.Assign(cfg.AgentLeft, PlayerSide.Left, cfg.SkillSetLeft, false);
            BattleManager.Instance.BotManager.Assign(cfg.AgentRight, PlayerSide.Right, cfg.SkillSetRight, false);
        }

        private List<BattleConfig> GenerateConfigs(List<Bot> agents)
        {
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



        private void SaveCheckpoint(SimulationCheckpoint checkpoint)
        {
            string folder = Path.Combine(Application.persistentDataPath, "Settings");
            string path = $"{folder}/simulation_checkpoint.json";
            string json = JsonUtility.ToJson(checkpoint, true);
            File.WriteAllText(path, json);
        }

        private SimulationCheckpoint LoadCheckpoint()
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

                Logger.Info($"[Checkpoint] Created checkpoint ID: {checkpoint.ID} at {checkpoint.CreatedAt}", true);
            }
            else
            {
                // Check if the configuration has changed
                bool configurationChanged = !Setting.IsConfigurationEqual(checkpoint.Setting);

                if (configurationChanged)
                {
                    Logger.Info("[Checkpoint] Configuration has changed. Resetting checkpoint.", true);
                    Logger.Info($"[Checkpoint] Old Config: Agents={checkpoint.Setting.SelectedAgents?.Length ?? 0}, " +
                              $"Timers={checkpoint.Setting.Timers?.Length ?? 0}, " +
                              $"Intervals={checkpoint.Setting.ActionIntervals?.Length ?? 0}, " +
                              $"Rounds={checkpoint.Setting.RoundSystem?.Length ?? 0}, " +
                              $"Skills={checkpoint.Setting.Skills?.Length ?? 0}, " +
                              $"Iteration={checkpoint.Setting.Iteration}", true);
                    Logger.Info($"[Checkpoint] New Config: Agents={Setting.SelectedAgents?.Length ?? 0}, " +
                              $"Timers={Setting.Timers?.Length ?? 0}, " +
                              $"Intervals={Setting.ActionIntervals?.Length ?? 0}, " +
                              $"Rounds={Setting.RoundSystem?.Length ?? 0}, " +
                              $"Skills={Setting.Skills?.Length ?? 0}, " +
                              $"Iteration={Setting.Iteration}", true);

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
            return new string[]{
                checkpoint.ID,
                $"{cfg.AgentLeft.ID}_vs_{cfg.AgentRight.ID}",
                $"Timer_{cfg.Timer}__ActInterval_{cfg.ActionInterval}__Round_{cfg.RoundSystem}__SkillLeft_{cfg.SkillSetLeft}__SkillRight_{cfg.SkillSetRight}",
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
    }
}

