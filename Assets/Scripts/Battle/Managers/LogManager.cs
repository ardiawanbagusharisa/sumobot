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
        Left,
        Right,
    }
    public enum PeriodicState
    {
        Start,
        Continues,
        End,
    }

    public class LogManager : MonoBehaviour
    {
        #region Log structures properties
        [Serializable]
        public class BattleLog
        {
            public string BattleID;
            public int CreatedAt;
            public string InputType;
            public float BattleTime;
            public float CountdownTime;
            public int RoundType;
            public int SimulationAmount;
            public float SimulationTimeScale;
            public PlayerStats LeftPlayerStats = new();
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
            public float Duration;
            public List<EventLog> PlayerEvents = new();
            public List<EventLog> StateEvents = new();
        }

        [Serializable]
        public class EventLog
        {
            public float StartedAt;
            public float UpdatedAt;
            public string Actor;
            public string Target;
            public float Duration;
            public string Category;
            public PeriodicState State;

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

        public static BattleLog Log;
        private static string logFolderPath;

        public static int CurrentGameIndex => Log.Games.Count > 0 ? Log.Games[^1].Index : 0;
        #endregion

        #region Action Logging methods

        public static void RegisterAction()
        {

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
                    logs.Add(action, new EventLogger(controller));
                }
                return logs;
            }
        }

        // Some actions maybe still hanging when the state is already ended, (e.g. dash and skill).
        // Therefore, we need manually add to stack
        public static void FlushActionLog(
            PlayerSide? side = null,
            ISumoAction action = null,
            bool isContinues = false)
        {
            if (side != null && action != null)
            {
                // Stop specific [action] from specific [side]
                if (ActionLoggers[(PlayerSide)side].TryGetValue(action.Type, out EventLogger actionLogger))
                {
                    if (isContinues)
                        actionLogger.Kill();
                    else
                        actionLogger.ForceStopAndSave();
                }
            }
            else if (side != null)
            {
                // Stop all actions from specific [side]
                foreach (EventLogger actionLogger in ActionLoggers[(PlayerSide)side].Values)
                {
                    actionLogger.ForceStopAndSave();
                }
            }
            else
            {
                // Stop all actions from all sides
                foreach (EventLogger actionLogger in ActionLoggers[PlayerSide.Left].Values)
                {
                    actionLogger.ForceStopAndSave();
                }
                foreach (EventLogger actionLogger in ActionLoggers[PlayerSide.Right].Values)
                {
                    actionLogger.ForceStopAndSave();
                }
            }
        }

        public static void UpdateActionLog(PlayerSide side)
        {
            BattleState currentState = BattleManager.Instance.CurrentState;
            if (currentState == BattleState.Battle_Ongoing || currentState == BattleState.Battle_End)
                foreach (EventLogger action in ActionLoggers[side].Values)
                {
                    action.Update();
                }
        }

        public static void CallActionLog(
            PlayerSide side,
            ISumoAction action,
            PeriodicState state = PeriodicState.Start)
        {
            if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
            {
                EventLogger actionLog = ActionLoggers[side][action.Type];
                actionLog.Call(action, state);
            }
        }
        #endregion

        #region Core Battle Log methods

        public static void InitLog()
        {
            string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string folderName = $"battle_{timestamp}";

            logFolderPath = Path.Combine(Application.persistentDataPath, "Logs", folderName);
            Directory.CreateDirectory(logFolderPath);
        }

        public static void InitBattle(int simAmount = 0, float simTimeScale = 0)
        {
            BattleManager battleManager = BattleManager.Instance;

            Log = new()
            {
                InputType = battleManager.BattleInputType.ToString(),
                BattleID = battleManager.Battle.BattleID.ToString(),
                CountdownTime = battleManager.CountdownTime,
                BattleTime = battleManager.BattleTime,
                RoundType = (int)battleManager.RoundSystem,
                SimulationAmount = simAmount,
                SimulationTimeScale = simTimeScale,
                CreatedAt = (int)DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            };

            SaveBattle();
        }

        public static void SetPlayerBots(Bot left, Bot right)
        {
            Log.LeftPlayerStats.Bot = left.ID;
            Log.RightPlayerStats.Bot = right.ID;
        }

        public static void UpdateMetadata(bool logTakenAction = true)
        {
            Log.LeftPlayerStats.SkillType = BattleManager.Instance.Battle.LeftPlayer.Skill.Type.ToString();
            Log.RightPlayerStats.SkillType = BattleManager.Instance.Battle.RightPlayer.Skill.Type.ToString();

            if (logTakenAction && Log.Games.Count > 0)
            {
                RoundLog currRound = GetCurrentRound();
                if (currRound != null)
                {
                    Log.LeftPlayerStats.ActionTaken += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Left" && x.Category == "Action" && (x.State != PeriodicState.End)).Count();

                    Log.RightPlayerStats.ActionTaken += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Right" && x.Category == "Action" && (x.State != PeriodicState.End)).Count();

                    Log.LeftPlayerStats.ContactMade += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Left" && x.Category == "Collision" && (x.State != PeriodicState.End)).Count();

                    Log.RightPlayerStats.ContactMade += GetCurrentRound().PlayerEvents.FindAll((x) => x.Actor == "Right" && x.Category == "Collision" && (x.State != PeriodicState.End)).Count();
                }
            }

            SaveBattle();
        }

        public static void StartGameLog()
        {
            int newGameIndex = CurrentGameIndex;

            if (Log.Games.Count > 0)
                newGameIndex += 1;

            GameLog newGame = new()
            {
                Index = newGameIndex,
                Timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
            };
            Log.Games.Add(newGame);
        }

        public static void StartRound(int index)
        {
            RoundLog newRound = new()
            {
                Index = index,
                Timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
            };
            Log.Games[CurrentGameIndex].Rounds.Add(newRound);
        }

        public static void SetRoundWinner(string winner)
        {
            Debug.Log($"Setting eventLog winner {winner}");
            RoundLog round = GetCurrentRound();
            if (round != null)
            {
                round.Winner = winner;
                round.Duration = BattleManager.Instance.ElapsedTime;

                if (round.Winner == "Left")
                {
                    Log.LeftPlayerStats.WinPerRound += 1;
                }
                else if (round.Winner == "Right")
                {
                    Log.RightPlayerStats.WinPerRound += 1;
                }
            }
        }

        public static void SetGameWinner(BattleWinner winner)
        {
            Log.Games[CurrentGameIndex].Winner = winner.ToString();
            if (winner == BattleWinner.Left)
                Log.LeftPlayerStats.WinPerGame += 1;
            else if (winner == BattleWinner.Right)
                Log.RightPlayerStats.WinPerGame += 1;
        }

        public static void LogBattleState(
            bool includeInCurrentRound = false,
            Dictionary<string, object> data = null)
        {
            if (!includeInCurrentRound)
            {
                Log.Events.Add(new EventLog
                {
                    StartedAt = Time.time,
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
                        StartedAt = BattleManager.Instance.ElapsedTime,
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
            PeriodicState state = PeriodicState.Start,
            float? startedAt = null,
            float? updatedAt = null,
            Dictionary<string, object> data = null)
        {
            RoundLog roundLog = GetCurrentRound();
            if (roundLog != null)
            {
                EventLog eventLog = new()
                {
                    Actor = actor.ToString(),
                    Target = target.ToString() ?? null,
                    Data = data,
                    State = state,
                    Category = category,
                    StartedAt = startedAt != null ? (float)startedAt : BattleManager.Instance.ElapsedTime
                };
                eventLog.UpdatedAt = updatedAt != null ? (float)updatedAt : eventLog.UpdatedAt = BattleManager.Instance.ElapsedTime;
                eventLog.Duration = eventLog.UpdatedAt - eventLog.StartedAt;
                roundLog.PlayerEvents.Add(eventLog);
            }
        }
        public static RoundLog GetCurrentRound()
        {
            if (Log.Games[CurrentGameIndex].Rounds.Count == 0)
            {
                Debug.LogWarning("No eventLog started yet.");
                return null;
            }

            return Log.Games[CurrentGameIndex].Rounds[^1];
        }
        #endregion

        #region Saving methods
        public static void SaveCurrentGame()
        {

            string json = JsonConvert.SerializeObject(Log.Games[CurrentGameIndex], Formatting.Indented);
            string paddedIndex = CurrentGameIndex.ToString("D3"); // D3 = 3-digit padding
            string savePath = Path.Combine(logFolderPath, $"game_{paddedIndex}.json");
            File.WriteAllText(savePath, json);
        }

        public static void SaveBattle()
        {
            string json = JsonConvert.SerializeObject(Log, Formatting.Indented);
            string savePath = Path.Combine(logFolderPath, "metadata.json");
            File.WriteAllText(savePath, json);
        }

        public static void SortAndSave()
        {
            Log.Games[CurrentGameIndex].Rounds.ForEach((rounds) =>
            {
                rounds.PlayerEvents = rounds.PlayerEvents.OrderBy(log => log?.UpdatedAt).ToList();
            });
            SaveCurrentGame();
        }

        public static void LogLastPosition()
        {
            RoundLog roundLog = GetCurrentRound();
            if (roundLog != null)
            {
                Battle battle = BattleManager.Instance.Battle;

                EventLogger leftLog = new(battle.LeftPlayer, isAction: false);
                EventLogger rightLog = new(battle.RightPlayer, isAction: false);

                leftLog.Save("LastPosition", roundLog.Duration);
                rightLog.Save("LastPosition", roundLog.Duration);
            }

        }
        #endregion
    }

}