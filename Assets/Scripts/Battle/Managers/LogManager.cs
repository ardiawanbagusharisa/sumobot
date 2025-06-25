using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CoreSumo;
using Newtonsoft.Json;
using UnityEngine;

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
    private class BattleLog
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
    private class GameLog
    {
        public int Index;
        public string Timestamp;
        public string Winner;
        public List<RoundLog> Rounds = new();
    }

    [Serializable]
    private class RoundLog
    {
        public int Index;
        public string Timestamp;
        public string Winner;
        public List<EventLog> ActionEvents = new();
        public List<EventLog> StateEvents = new();
    }

    [Serializable]
    private class EventLog
    {
        public string LoggedAt;
        public string StartedAt;
        public string UpdatedAt;
        public string Actor;
        public string Target;
        public string Category;
        public bool IsStart;

        public Dictionary<string, object> Data = new Dictionary<string, object>();
    }

    [Serializable]
    private class PlayerStats
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
    public static Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>> ActionLoggers = new Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>>();

    private static BattleLog battleLog;
    private static string logFolderPath;

    public static int CurrentGameIndex => battleLog.Games.Count > 0 ? battleLog.Games[^1].Index : 0;
    #endregion

    #region Action Logging methods

    public static void SetPlayerAction()
    {
        SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
        SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;

        if (!ActionLoggers.ContainsKey(leftPlayer.Side))
            ActionLoggers.Add(leftPlayer.Side, InitByController(leftPlayer));
        if (!ActionLoggers.ContainsKey(rightPlayer.Side))
            ActionLoggers.Add(rightPlayer.Side, InitByController(rightPlayer));

        static Dictionary<string, DebouncedLogger> InitByController(SumoController controller)
        {
            Dictionary<string, DebouncedLogger> logs = new();
            IEnumerable<ActionType> actionTypes = Enum.GetValues(typeof(ActionType)).Cast<ActionType>();
            foreach (ActionType action in actionTypes)
            {
                switch (action)
                {
                    case ActionType.Dash:
                        logs.Add(action.ToString(), new DebouncedLogger(controller, controller.DashDuration));
                        break;
                    case ActionType.SkillStone:
                    case ActionType.SkillBoost:
                        logs.Add(action.ToString(), new DebouncedLogger(controller, controller.Skill.TotalDuration));
                        break;
                    default:
                        logs.Add(action.ToString(), new DebouncedLogger(controller, 0.1f));
                        break;
                }
            }
            return logs;
        }
    }

    // Some actions maybe still hanging when the state is already ended, (e.g. dash and skill).
    // Therefore, we need manually add to stack
    public static void CleanIncompletePlayerAction()
    {
        foreach (Dictionary<string, DebouncedLogger> actionSide in ActionLoggers.Values)
        {
            foreach (DebouncedLogger actionLogger in actionSide.Values)
            {
                if (actionLogger.IsActive)
                    actionLogger.ForceStopAndSave();
            }
        }
    }

    public static void UpdatePlayerActionLog(PlayerSide side)
    {
        if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing || BattleManager.Instance.CurrentState == BattleState.Battle_End)
            foreach (var action in ActionLoggers[side].Values)
            {
                action.Update();
            }
    }

    public static void CallPlayerActionLog(PlayerSide side, ISumoAction action)
    {
        if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
        {
            var actionLog = ActionLoggers[side][action.Type.ToString()];
            actionLog.Call(action);
        }
    }
    #endregion

    #region Core Battle Log methods

    public static void InitLog()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var folderName = $"battle_{timestamp}";

        logFolderPath = Path.Combine(Application.persistentDataPath, "Logs", folderName);
        Directory.CreateDirectory(logFolderPath);
    }

    public static void InitBattle()
    {
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
            var leftBot = BattleManager.Instance.Bot.Left;
            var rightBot = BattleManager.Instance.Bot.Right;

            if (leftBot != null)
            {
                battleLog.LeftPLayerStats.Bot = leftBot?.ID ?? "";
            }

            if (rightBot != null)
            {
                battleLog.RightPlayerStats.Bot = rightBot?.ID ?? "";
            }
        }

        battleLog.LeftPLayerStats.SkillType = BattleManager.Instance.Battle.LeftPlayer.Skill.Type.ToString();
        battleLog.RightPlayerStats.SkillType = BattleManager.Instance.Battle.RightPlayer.Skill.Type.ToString();

        if (battleLog.Games.Count > 0)
        {
            RoundLog currRound = GetCurrentRound();
            if (currRound != null)
            {
                battleLog.LeftPLayerStats.ActionTaken += GetCurrentRound().ActionEvents.FindAll((x) => x.Actor == "Left" && (string)x.Data["Type"] != "Bounce").Count();
                battleLog.RightPlayerStats.ActionTaken += GetCurrentRound().ActionEvents.FindAll((x) => x.Actor == "Right" && (string)x.Data["Type"] != "Bounce").Count();
                battleLog.LeftPLayerStats.ContactMade += GetCurrentRound().ActionEvents.FindAll((x) => x.Actor == "Left" && (string)x.Data["Type"] == "Bounce").Count();
                battleLog.RightPlayerStats.ContactMade += GetCurrentRound().ActionEvents.FindAll((x) => x.Actor == "Right" && (string)x.Data["Type"] == "Bounce").Count();
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
                LoggedAt = DateTime.Now.ToString("o"),
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
        string category = "action",
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

            eventLog.StartedAt = startedAt != null ? startedAt.ToString() : BattleManager.Instance.ElapsedTime.ToString();
            eventLog.UpdatedAt = updatedAt != null ? updatedAt.ToString() : eventLog.UpdatedAt = BattleManager.Instance.ElapsedTime.ToString();
            roundLog.ActionEvents.Add(eventLog);
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

        var json = JsonConvert.SerializeObject(battleLog.Games[CurrentGameIndex], Formatting.Indented);
        string paddedIndex = CurrentGameIndex.ToString("D3"); // D3 = 3-digit padding
        var savePath = Path.Combine(logFolderPath, $"game_{paddedIndex}.json");
        File.WriteAllText(savePath, json);
    }

    public static void SaveBattle()
    {
        var json = JsonConvert.SerializeObject(battleLog, Formatting.Indented);
        var savePath = Path.Combine(logFolderPath, "metadata.json");
        File.WriteAllText(savePath, json);
    }

    public static void SortAndSave()
    {
        battleLog.Games[CurrentGameIndex].Rounds.ForEach((rounds) =>
        {
            rounds.ActionEvents = rounds.ActionEvents.OrderBy(log => float.Parse(log?.StartedAt ?? "0")).ToList();
        });
        SaveCurrentGame();
    }
    #endregion
}
