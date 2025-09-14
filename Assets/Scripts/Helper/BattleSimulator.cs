using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SumoManager;
using SumoBot;
using System.Linq;
using SumoCore;
using System;
using System.IO;

namespace SumoHelper
{
    public class BattleSimulator : MonoBehaviour
    {
        public float DefaultTimeScale = 2f;
        public bool SimulationOnStart = false;

        [SerializeField, Tooltip("If True, every iteration will be saved, and can continue it later including saved SimulationSetting")]
        public bool UseCheckpoint = true;

        public int RoundCountdown = 3;
        public SimulationSetting Setting;

        private List<Bot> Agents = new();
        private List<BattleConfig> _configs;
        private int currentConfigIndex = 0;
        private int firstConfigIndex = 0;
        private SimulationCheckpoint checkpoint;

        void OnDisable()
        {
            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleStateChanged);
        }

        public void PrepareSimulation()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = true;
#endif

            if (UseCheckpoint)
                checkpoint = LoadCheckpoint();
            else
                checkpoint = new SimulationCheckpoint
                {
                    Setting = Setting,
                    Iteration = 1,
                };

            if (Setting.Timers.Length == 0)
                throw new Exception("Timers can't be empty");
            if (Setting.RoundSystem.Length == 0)
                throw new Exception("RoundSystem can't be empty");
            if (Setting.Iteration == 0)
                throw new Exception("Iteration must > 0");
            if (Setting.SelectedAgents.Length == 1)
                throw new Exception("SelectedAgents must > 1");

            SelectAgents();

            Application.runInBackground = true;
            BattleManager.Instance.CountdownTime = RoundCountdown;
            BattleManager.Instance.BotManager.LeftEnabled = true;
            BattleManager.Instance.BotManager.RightEnabled = true;

            _configs = GenerateConfigs(Agents);

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
                        if (Setting.ExcludedAgents.Length > 0)
                            if (!Setting.ExcludedAgents.Contains(botInstance.ID))
                                Agents.Add(botInstance);
                    }
                }
            }

            Debug.Log($"Loaded {Agents.Count}\nagents: {string.Join(", ", Agents.Select(a => a.ID))}");
        }

        private IEnumerator RunSimulations()
        {
            yield return new WaitForSeconds(0.5f);

            for (currentConfigIndex = checkpoint.ConfigIndex; currentConfigIndex < _configs.Count; currentConfigIndex++)
            {
                BattleConfig cfg = _configs[currentConfigIndex];

                if (currentConfigIndex > firstConfigIndex)
                {
                    ApplyConfig(_configs[currentConfigIndex]);
                }

                for (int iter = checkpoint.Iteration; iter <= cfg.Iteration; iter++)
                {
                    Debug.Log($"[Simulation] Config {currentConfigIndex + 1}/{_configs.Count}, Iteration {iter}/{cfg.Iteration} | " +
                              $"{cfg.AgentLeft.ID} vs {cfg.AgentRight.ID} | " +
                              $"Timer={cfg.Timer}, Interval={cfg.ActionInterval}, SkillLeft={cfg.SkillSetLeft}, SkillRight={cfg.SkillSetRight}");

                    yield return new WaitForSeconds(1);

                    if (SimulationOnStart || currentConfigIndex > 0 || iter > 1)
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
                    if (UseCheckpoint)
                        SaveCheckpoint(checkpoint);
                    yield return new WaitForEndOfFrame();
                }

                checkpoint.Iteration = 1;
            }

            Debug.Log("All simulations complete.");
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

            if (cfg.AgentLeft.UseAsync || cfg.AgentRight.UseAsync)
                Time.timeScale = 1f;
            else
                Time.timeScale = cfg.TimeScale;

            var path = new string[]{
                $"{cfg.AgentLeft.ID}_vs_{cfg.AgentRight.ID}",
                $"Timer_{cfg.Timer}__ActInterval_{cfg.ActionInterval}__Round_{cfg.RoundSystem}__SkillLeft_{cfg.SkillSetLeft}__SkillRight_{cfg.SkillSetRight}",
            };

            LogManager.UnregisterAction();
            LogManager.InitLog(path);
            LogManager.InitBattle(cfg);
            var newBattle = new Battle(Guid.NewGuid().ToString(), cfg.RoundSystem);
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

            Debug.Log($"Generated configs: {configs.Count}");
            Debug.Log($"Game will run {configs.Aggregate(0, (sum, cfg) => sum + cfg.Iteration)} matches in total.");
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
                checkpoint = JsonUtility.FromJson<SimulationCheckpoint>(json);
            }

            if (checkpoint == null)
            {
                checkpoint = new SimulationCheckpoint
                {
                    Setting = Setting,
                    Iteration = 1,
                };
            }
            else
            {
                if (UseCheckpoint)
                {
                    Setting = checkpoint.Setting;
                }
                currentConfigIndex = checkpoint.ConfigIndex;
                firstConfigIndex = checkpoint.ConfigIndex;
            }

            return checkpoint;
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
        public int[] Timers;
        public float[] ActionIntervals;

        public int Iteration;
        public RoundSystem[] RoundSystem;

        [SerializeField, Tooltip("If Skills is not specified, use skill from original script")]
        public SkillType[] Skills;

        [SerializeField, Tooltip("Run only these agents. If empty, run all scanned agents")]
        public string[] SelectedAgents = new string[] { };

        [SerializeField, Tooltip("Run agents except these ExcludedAgents. Will be ignored when SelectedAgents filled")]
        public string[] ExcludedAgents = new string[] { "Bot_Template", "Bot_LLM_ActionGPT", "Bot_SLM_ActionGPT", "Bot_ML_Classification" };
    }

    [Serializable]
    public class SimulationCheckpoint
    {
        public SimulationSetting Setting;
        public int TotalConfigs;
        public int ConfigIndex;
        public int Iteration;
    }
}
