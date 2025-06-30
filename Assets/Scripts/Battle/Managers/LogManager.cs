using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using SumoBot;
using SumoCore;
using SumoHelper;
using UnityEngine;

namespace SumoManager
{
    public enum LogActorType
    {
        System,
        LeftPlayer,
        RightPlayer,
    }

    public class LogManager
    {
        #region Log structures properties
        [Serializable]
        public class BattleLog
        {
            public string BattleID;
            public string InputType;
            public float BattleTime;
            public float CountdownTime;
            public int RoundType;
            public PlayerStats LeftPLayerStats = new();
            public PlayerStats RightPlayerStats = new();

            public List<EventLog> Events = new();

            [NonSerialized]
            public List<GameLog> Games = new();
        }

        [Serializable]
        public class GameLog
        {
            public int Index;
            public string Timestamp;
            public string Winner;
            public List<RoundLog> Rounds = new();
        }

        [Serializable]
        public class RoundLog
        {
            public int Index;
            public string Timestamp;
            public string Winner;
            public List<EventLog> PlayerEvents = new();
            public List<EventLog> StateEvents = new();
        }

        [Serializable]
        public class EventLog
        {
            public string LoggedAt;
            public float StartedAt;
            public float UpdatedAt;
            public string Actor;
            public string Target;
            public string Category;
            public bool IsStart;

            public Dictionary<string, object> Data = new();
        }

        [Serializable]
        public class PlayerStats
        {
            public string SkillType;
            public string Bot;
            public int WinPerGame = 0;
            public int WinPerRound = 0;
            public int ActionTaken = 0;
            public int ContactMade = 0;
        }
        #endregion

        #region class properties 
        public static Dictionary<PlayerSide, Dictionary<ActionType, EventLogger>> ActionLoggers = new();

        private static BattleLog battleLog;
        private static string logFolderPath;
        private static bool IsLogEnabled => ReplayManager.Instance == null;

        public static int CurrentGameIndex => battleLog.Games.Count > 0 ? battleLog.Games[^1].Index : 0;
        #endregion

        #region Action Logging methods

        public static void RegisterAction()
        {
            if (!IsLogEnabled)
            {
                return;
            }

            SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
            SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;

            if (!ActionLoggers.ContainsKey(leftPlayer.Side))
                ActionLoggers.Add(leftPlayer.Side, InitByController(leftPlayer));
            if (!ActionLoggers.ContainsKey(rightPlayer.Side))
                ActionLoggers.Add(rightPlayer.Side, InitByController(rightPlayer));

            static Dictionary<ActionType, EventLogger> InitByController(SumoController controller)
            {
                Dictionary<ActionType, EventLogger> logs = new();
                IEnumerable<ActionType> actionTypes = Enum.GetValues(typeof(ActionType)).Cast<ActionType>();
                foreach (ActionType action in actionTypes)
                {
                    switch (action)
                    {
                        case ActionType.Dash:
                            logs.Add(action, new EventLogger(controller, controller.DashDuration));
                            break;
                        case ActionType.SkillStone:
                        case ActionType.SkillBoost:
                            logs.Add(action, new EventLogger(controller, controller.Skill.TotalDuration));
                            break;
                        default:
                            logs.Add(action, new EventLogger(controller, 0.1f));
                            break;
                    }
                }
                return logs;
            }
        }

        // Some actions maybe still hanging when the state is already ended, (e.g. dash and skill).
        // Therefore, we need manually add to stack
        public static void FlushActionLog()
        {
            if (!IsLogEnabled)
            {
                return;
            }

            foreach (Dictionary<ActionType, EventLogger> actionSide in ActionLoggers.Values)
            {
                foreach (EventLogger actionLogger in actionSide.Values)
                {
                    if (actionLogger.IsActive && actionLogger.ForceSave)
                        actionLogger.ForceStopAndSave();
                }
            }
        }

        public static void UpdateActionLog(PlayerSide side)
        {
            if (!IsLogEnabled)
            {
                return;
            }

            BattleState currentState = BattleManager.Instance.CurrentState;
            if (currentState == BattleState.Battle_Ongoing || currentState == BattleState.Battle_End)
                foreach (EventLogger action in ActionLoggers[side].Values)
                {
                    action.Update();
                }
        }

        public static void CallActionLog(PlayerSide side, ISumoAction action)
        {
            if (!IsLogEnabled)
            {
                return;
            }

            if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
            {
                EventLogger actionLog = ActionLoggers[side][action.Type];
                actionLog.Call(action);
            }
        }
        #endregion

        #region Core Battle Log methods

        public static void InitLog()
        {
            if (!IsLogEnabled)
            {
                return;
            }

            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string folderName = $"battle_{timestamp}";

            logFolderPath = Path.Combine(Application.persistentDataPath, "Logs", folderName);
            Directory.CreateDirectory(logFolderPath);
        }

        public static void InitBattle()
        {
            if (!IsLogEnabled)
            {
                return;
            }

            BattleManager battleManager = BattleManager.Instance;

            battleLog = new()
            {
                InputType = battleManager.BattleInputType.ToString(),
                BattleID = battleManager.Battle.BattleID.ToString(),
                CountdownTime = battleManager.CountdownTime,
                BattleTime = battleManager.BattleTime,
                RoundType = (int)battleManager.RoundSystem
            };
            SaveBattle();
        }

        public static void UpdateMetadata()
        {
            if (BattleManager.Instance.Bot.IsEnable)
            {
                Bot leftBot = BattleManager.Instance.Bot.Left;
                Bot rightBot = BattleManager.Instance.Bot.Right;

                if (leftBot != null)
                    battleLog.LeftPLayerStats.Bot = leftBot?.ID ?? "";

                if (rightBot != null)
                    battleLog.RightPlayerStats.Bot = rightBot?.ID ?? "";
            }

            battleLog.LeftPLayerStats.SkillType = BattleManager.Instance.Battle.LeftPlayer.Skill.Type.ToString();
            battleLog.RightPlayerStats.SkillType = BattleManager.Instance.Battle.RightPlayer.Skill.Type.ToString();

            if (battleLog.Games.Count > 0)
            {
                RoundLog currRound = GetCurrentRound();
                if (currRound != null)
                {
                    battleLog.LeftPLayerStats.ActionTaken += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Left" && x.Category == "Action").Count();
                    battleLog.RightPlayerStats.ActionTaken += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Right" && x.Category == "Action").Count();

                    battleLog.LeftPLayerStats.ContactMade += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Left" && x.Category == "Collision").Count();
                    battleLog.RightPlayerStats.ContactMade += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Right" && x.Category == "Collision").Count();
                }
            }

            SaveBattle();
        }

        public static void StartGameLog()
        {
            int newGameIndex = CurrentGameIndex;

            if (battleLog.Games.Count > 0)
                newGameIndex += 1;

            GameLog newGame = new()
            {
                Index = newGameIndex,
                Timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
            };
            battleLog.Games.Add(newGame);
        }

        public static void StartRound(int index)
        {
            RoundLog newRound = new()
            {
                Index = index,
                Timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
            };
            battleLog.Games[CurrentGameIndex].Rounds.Add(newRound);
        }

        public static void SetRoundWinner(string winner)
        {
            Debug.Log($"Setting eventLog winner {winner}");
            RoundLog round = GetCurrentRound();
            if (round != null)
            {
                round.Winner = winner;
                if (round.Winner == "Left")
                {
                    battleLog.LeftPLayerStats.WinPerRound += 1;
                }
                else if (round.Winner == "Right")
                {
                    battleLog.RightPlayerStats.WinPerRound += 1;
                }
            }
        }

        public static void SetGameWinner(BattleWinner winner)
        {
            battleLog.Games[CurrentGameIndex].Winner = winner.ToString();
            if (winner == BattleWinner.Left)
                battleLog.LeftPLayerStats.WinPerGame += 1;
            else if (winner == BattleWinner.Right)
                battleLog.RightPlayerStats.WinPerGame += 1;
        }

        public static void LogBattleState(
            bool includeInCurrentRound = false,
            Dictionary<string, object> data = null)
        {
            if (!includeInCurrentRound)
            {
                battleLog.Events.Add(new EventLog
                {
                    LoggedAt = DateTime.Now.ToString(),
                    Actor = "System",
                    Data = data
                });
            }
            else
            {
                RoundLog roundLog = GetCurrentRound();
                if (roundLog != null)
                {
                    EventLog eventLog = new()
                    {
                        LoggedAt = BattleManager.Instance.ElapsedTime.ToString(),
                        Actor = "System",
                        Data = data,
                    };
                    roundLog.StateEvents.Add(eventLog);
                }
            }
        }

        public static void LogPlayerEvents(
            PlayerSide actor,
            PlayerSide? target = null,
            string category = "Action",
            bool isStart = false,
            float? startedAt = null,
            float? updatedAt = null,
            Dictionary<string, object> data = null)
        {
            RoundLog roundLog = GetCurrentRound();
            if (roundLog != null)
            {
                EventLog eventLog = new()
                {
                    LoggedAt = BattleManager.Instance.ElapsedTime.ToString(),
                    Actor = actor.ToString(),
                    Target = target.ToString() ?? null,
                    Data = data,
                    IsStart = isStart,
                    Category = category,
                };

                eventLog.StartedAt = startedAt != null ? (float)startedAt : BattleManager.Instance.ElapsedTime;
                eventLog.UpdatedAt = updatedAt != null ? (float)updatedAt : eventLog.UpdatedAt = BattleManager.Instance.ElapsedTime;
                roundLog.PlayerEvents.Add(eventLog);
            }
        }
        private static RoundLog GetCurrentRound()
        {
            if (battleLog.Games[CurrentGameIndex].Rounds.Count == 0)
            {
                Debug.LogWarning("No eventLog started yet.");
                return null;
            }

            return battleLog.Games[CurrentGameIndex].Rounds[^1];
        }
        #endregion

        #region Saving methods
        public static void SaveCurrentGame()
        {

            string json = JsonConvert.SerializeObject(battleLog.Games[CurrentGameIndex], Formatting.Indented);
            string paddedIndex = CurrentGameIndex.ToString("D3"); // D3 = 3-digit padding
            string savePath = Path.Combine(logFolderPath, $"game_{paddedIndex}.json");
            File.WriteAllText(savePath, json);
        }

        public static void SaveBattle()
        {
            string json = JsonConvert.SerializeObject(battleLog, Formatting.Indented);
            string savePath = Path.Combine(logFolderPath, "metadata.json");
            File.WriteAllText(savePath, json);
        }

        public static void SortAndSave()
        {
            battleLog.Games[CurrentGameIndex].Rounds.ForEach((rounds) =>
            {
                rounds.PlayerEvents = rounds.PlayerEvents.OrderBy(log => log?.UpdatedAt).ToList();
            });
            SaveCurrentGame();
        }
        #endregion
    }

}