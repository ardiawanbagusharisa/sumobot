using System;
using System.Collections.Generic;
using System.IO;
using BattleLoop;
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
    private static BattleLog _battleLog;
    private static string _logFilePath;

    [Serializable]
    private class BattleLog
    {
        public string battle_id;
        public string input_type;
        public List<EventLog> events = new();
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
        public string timestamp;
        public string battle_timestamp;
        public string actor;
        public string target;

        public Dictionary<string, object> data = new Dictionary<string, object>();
    }

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

    public static void StartRound(int index)
    {
        _battleLog.rounds.Add(new RoundLog { index = index });
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

    public static void LogGlobalEvent(
        LogActorType actor,
        LogActorType? target = null,
        Dictionary<string, object> data = null)
    {
        _battleLog.events.Add(new EventLog
        {
            timestamp = DateTime.Now.ToString("o"),
            actor = actor.ToString(),
            target = target.ToString() ?? null,
            data = data
        });
        SaveLog();
    }

    public static void LogRoundEvent(
        LogActorType actor,
        LogActorType? target = null,
        Dictionary<string, object> data = null)
    {
        var round = GetCurrentRound();
        if (round != null)
        {
            round.events.Add(new EventLog
            {
                timestamp = DateTime.Now.ToString("o"),
                battle_timestamp = BattleManager.Instance.ElapsedTime.ToString(),
                actor = actor.ToString(),
                target = target.ToString() ?? null,
                data = data,
            });
            SaveLog();
        }
    }

    private static RoundLog GetCurrentRound()
    {
        if (_battleLog.rounds.Count == 0)
        {
            Debug.LogWarning("No round started yet.");
            return null;
        }

        return _battleLog.rounds[^1]; // last round
    }

    private static void SaveLog()
    {
        string json = JsonConvert.SerializeObject(_battleLog, Formatting.Indented);
        File.WriteAllText(_logFilePath, json);
    }
}
