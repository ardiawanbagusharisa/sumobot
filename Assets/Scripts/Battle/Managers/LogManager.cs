using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using CoreSumo;
using Newtonsoft.Json;
using Unity.VisualScripting;
using UnityEngine;


public enum LogActorType
{
    System,
    LeftPlayer,
    RightPlayer,
}

public class LogManager
{
    public static Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>> ActionLoggers = new Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>>();

    private static BattleLog _battleLog;
    private static string _logFilePath;

    public static int CurrentGameIndex => _battleLog.games.Count > 0 ? _battleLog.games[^1].index : 0;

    [Serializable]
    private class BattleLog
    {
        public string battle_id;
        public string input_type;

        public List<EventLog> events = new();
        public List<GameLog> games = new();
    }

    [Serializable]
    private class GameLog
    {
        public int index;
        public string timestamp;
        public string winner = "";
        public List<RoundLog> rounds = new();
    }

    [Serializable]
    private class RoundLog
    {
        public int index;
        public string winner = "";
        public List<EventLog> events = new();
    }

    [Serializable]
    private class EventLog
    {
        public string loggedAt;
        public string startedAt;
        public string updatedAt;
        public string actor;
        public string target;

        public Dictionary<string, object> data = new Dictionary<string, object>();
    }

    #region Sumo Action Logging

    public static void SetPlayerAction()
    {
        var leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
        var rightPlayer = BattleManager.Instance.Battle.RightPlayer;
        if (!ActionLoggers.ContainsKey(leftPlayer.Side))
            ActionLoggers.Add(leftPlayer.Side, InitByController(leftPlayer));
        if (!ActionLoggers.ContainsKey(rightPlayer.Side))
            ActionLoggers.Add(rightPlayer.Side, InitByController(rightPlayer));

        static Dictionary<string, DebouncedLogger> InitByController(SumoController controller)
        {
            return new Dictionary<string, DebouncedLogger>
            {
                {"Accelerate", new DebouncedLogger(controller, 0.1f) },

                // it's enough to assign TurnLeftAngle, TurnRightAngle, TurnAngle to single action
                { "TurnRight", new DebouncedLogger(controller, 0.1f) },
                { "TurnLeft", new DebouncedLogger(controller, 0.1f) },

                { "Dash", new DebouncedLogger(controller, controller.DashDuration) },
                {"Skill", new DebouncedLogger(controller, controller.Skill.SkillDuration) }
            };
        }
    }

    // Some actions maybe still hanging when the state is already ended, (e.g. dash and skill).
    // Therefore, we need manually add to stack
    public static void CleanIncompletePlayerAction()
    {
        foreach (Dictionary<string, DebouncedLogger> actionSide in ActionLoggers.Values)
        {
            foreach (DebouncedLogger action in actionSide.Values)
            {
                if (action.IsActive)
                {
                    action.SaveToLog();
                }
            }
        }
    }

    public static void UpdatePlayerActionLog(PlayerSide side)
    {
        if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
            foreach (var action in ActionLoggers[side].Values)
            {
                action.Update();
            }
    }

    public static void CallPlayerActionLog(PlayerSide side, string logName, string actionName = null, string param = null)
    {
        var actionLog = ActionLoggers[side][logName];
        actionLog.Call(actionName ?? logName, param);
    }
    #endregion

    #region Core Battle Log

    public static void InitBattle()
    {
        string input = BattleManager.Instance.BattleInputType.ToString();
        string battleId = BattleManager.Instance.Battle.BattleID;
        _battleLog = new BattleLog { battle_id = battleId, input_type = input };

        string folder = Path.Combine(Application.persistentDataPath, "Logs");
        Directory.CreateDirectory(folder);

        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        _logFilePath = Path.Combine(folder, $"battle_{timestamp}.json");

        SaveLog();
    }

    public static void StartNewGame()
    {
        int newGameIndex = CurrentGameIndex;
        if (_battleLog.games.Count > 0)
        {
            newGameIndex += 1;
        }
        _battleLog.games.Add(new GameLog { index = newGameIndex });
        SaveLog();
    }

    public static void StartRound(int index)
    {
        _battleLog.games[CurrentGameIndex].rounds.Add(new RoundLog { index = index });
        SaveLog();
    }

    public static void SetRoundWinner(LogActorType winner)
    {
        var round = GetCurrentRound();
        if (round != null)
        {
            round.winner = winner.ToString();
            SaveLog();
        }
    }

    public static void SetGameWinner(BattleWinner winner)
    {
        _battleLog.games[CurrentGameIndex].winner = winner.ToString();
        SaveLog();
    }

    public static void LogGlobalEvent(
        LogActorType actor,
        LogActorType? target = null,
        Dictionary<string, object> data = null)
    {
        _battleLog.events.Add(new EventLog
        {
            loggedAt = DateTime.Now.ToString("o"),
            actor = actor.ToString(),
            target = target.ToString() ?? null,
            data = data
        });
        SaveLog();
    }

    public static void LogRoundEvent(
        LogActorType actor,
        LogActorType? target = null,
        float? startedAt = null,
        float? updatedAt = null,
        Dictionary<string, object> data = null)
    {
        var round = GetCurrentRound();
        if (round != null)
        {
            EventLog roundLog = new()
            {
                loggedAt = BattleManager.Instance.ElapsedTime.ToString(),
                actor = actor.ToString(),
                target = target.ToString() ?? null,
                data = data,
            };
            if (startedAt != null)
            {
                roundLog.startedAt = startedAt.ToString();
            }
            else
            {
                roundLog.startedAt = BattleManager.Instance.ElapsedTime.ToString();
            }

            if (updatedAt != null)
            {
                roundLog.updatedAt = updatedAt.ToString();
            }
            else
            {
                roundLog.updatedAt = BattleManager.Instance.ElapsedTime.ToString();
            }

            round.events.Add(roundLog);
            SaveLog();
        }
    }

    private static RoundLog GetCurrentRound()
    {
        if (_battleLog.games[CurrentGameIndex].rounds.Count == 0)
        {
            Debug.LogWarning("No round started yet.");
            return null;
        }

        return _battleLog.games[CurrentGameIndex].rounds[^1]; // last round
    }
    #endregion

    private static void SaveLog()
    {
        _battleLog.games.ForEach((game) =>
        {
            game.rounds.ForEach((rounds) =>
            {
                rounds.events = rounds.events.OrderBy(log => float.Parse(log.startedAt)).ToList();
            });
        });

        string json = JsonConvert.SerializeObject(_battleLog, Formatting.Indented);
        File.WriteAllText(_logFilePath, json);
    }

}
