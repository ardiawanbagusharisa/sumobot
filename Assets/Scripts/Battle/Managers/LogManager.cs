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
        public string battle_id;
        public string input_type;
        public float battle_time;
        public float countdown_time;
        public int round_type;
        public PlayerStats left_player_stats = new();
        public PlayerStats right_player_stats = new();

        public List<EventLog> events = new();

        [NonSerialized]
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
        public string timestamp;
        public string winner = "";
        public List<EventLog> action_events = new();
        public List<EventLog> state_events = new();
    }

    [Serializable]
    private class EventLog
    {
        public string logged_at;
        public string started_at;
        public string updated_at;
        public string actor;
        public string target;

        public Dictionary<string, object> data = new Dictionary<string, object>();
    }

    [Serializable]
    private class PlayerStats
    {
        public string skill_type;
        public string bot;
        public int win_per_game = 0;
        public int win_per_round = 0;
        public int actions_taken = 0;
        public int contact_made = 0;
    }
    #endregion

    #region class properties 
    public static Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>> ActionLoggers = new Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>>();

    private static BattleLog _battleLog;
    private static string _logFolderPath;

    public static int CurrentGameIndex => _battleLog.games.Count > 0 ? _battleLog.games[^1].index : 0;
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
            return new Dictionary<string, DebouncedLogger>
            {
                {"Accelerate", new DebouncedLogger(controller, 0.1f) },
                { "TurnRight", new DebouncedLogger(controller, 0.1f) },
                { "TurnLeft", new DebouncedLogger(controller, 0.1f) },
                { "TurnLeftAngle", new DebouncedLogger(controller, 0.1f) },
                { "TurnRightAngle", new DebouncedLogger(controller, 0.1f) },
                { "TurnAngle", new DebouncedLogger(controller, 0.1f) },
                { "Dash", new DebouncedLogger(controller, controller.DashDuration) },
                { "Skill", new DebouncedLogger(controller, controller.Skill.SkillDuration) }
            };
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
            var actionLog = ActionLoggers[side][action.Name];
            actionLog.Call(action.Name, action?.Param?.ToString(), reason: action.Reason);
        }
    }
    #endregion

    #region Core Battle Log methods

    public static void InitLog()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var folderName = $"battle_{timestamp}";

        _logFolderPath = Path.Combine(Application.persistentDataPath, "Logs", folderName);
        Directory.CreateDirectory(_logFolderPath);
    }

    public static void InitBattle()
    {
        BattleManager battleManager = BattleManager.Instance;

        _battleLog = new BattleLog();
        _battleLog.input_type = battleManager.BattleInputType.ToString();
        _battleLog.battle_id = battleManager.Battle.BattleID.ToString();
        _battleLog.countdown_time = battleManager.CountdownTime;
        _battleLog.battle_time = battleManager.BattleTime;
        _battleLog.round_type = (int)battleManager.RoundSystem;
        SaveBattle();
    }

    public static void UpdateMetadata()
    {
        var leftBot = BattleManager.Instance.Battle.LeftPlayer.gameObject.GetComponents<Bot>().FirstOrDefault(x => x.enabled);
        _battleLog.left_player_stats.bot = leftBot != null ? leftBot.Name() : "";

        var rightBot = BattleManager.Instance.Battle.RightPlayer.gameObject.GetComponents<Bot>().FirstOrDefault(x => x.enabled);
        _battleLog.right_player_stats.bot = rightBot != null ? rightBot.Name() : "";

        _battleLog.left_player_stats.skill_type = BattleManager.Instance.Battle.LeftPlayer.Skill.Type.ToString();
        _battleLog.right_player_stats.skill_type = BattleManager.Instance.Battle.RightPlayer.Skill.Type.ToString();

        if (_battleLog.games.Count > 0)
        {
            RoundLog currRound = GetCurrentRound();
            if (currRound != null)
            {
                _battleLog.left_player_stats.actions_taken += GetCurrentRound().action_events.FindAll((x) => x.actor == "Left" && (string)x.data["type"] != "Bounce").Count();
                _battleLog.right_player_stats.actions_taken += GetCurrentRound().action_events.FindAll((x) => x.actor == "Right" && (string)x.data["type"] != "Bounce").Count();
                _battleLog.left_player_stats.contact_made += GetCurrentRound().action_events.FindAll((x) => x.actor == "Left" && (string)x.data["type"] == "Bounce").Count();
                _battleLog.right_player_stats.contact_made += GetCurrentRound().action_events.FindAll((x) => x.actor == "Right" && (string)x.data["type"] == "Bounce").Count();
            }
        }

        SaveBattle();
    }

    public static void StartGameLog()
    {
        int newGameIndex = CurrentGameIndex;

        if (_battleLog.games.Count > 0)
            newGameIndex += 1;

        GameLog newGame = new()
        {
            index = newGameIndex,
            timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
        };
        _battleLog.games.Add(newGame);
    }

    public static void StartRound(int index)
    {
        RoundLog newRound = new()
        {
            index = index,
            timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss")
        };
        _battleLog.games[CurrentGameIndex].rounds.Add(newRound);
    }

    public static void SetRoundWinner(string winner)
    {
        Debug.Log($"Setting eventLog winner {winner}");
        RoundLog round = GetCurrentRound();
        if (round != null)
        {
            round.winner = winner;
            if (round.winner == "Left")
            {
                _battleLog.left_player_stats.win_per_round += 1;
            }
            else if (round.winner == "Right")
            {
                _battleLog.right_player_stats.win_per_round += 1;
            }
        }
    }

    public static void SetGameWinner(BattleWinner winner)
    {
        _battleLog.games[CurrentGameIndex].winner = winner.ToString();
        if (winner == BattleWinner.Left)
            _battleLog.left_player_stats.win_per_game += 1;
        else if (winner == BattleWinner.Right)
            _battleLog.right_player_stats.win_per_game += 1;
    }

    public static void LogBattleState(
        LogActorType actor,
        bool includeInCurrentRound = false,
        LogActorType? target = null,
        Dictionary<string, object> data = null)
    {
        if (!includeInCurrentRound)
        {
            _battleLog.events.Add(new EventLog
            {
                logged_at = DateTime.Now.ToString("o"),
                actor = actor.ToString(),
                target = target.ToString() ?? null,
                data = data
            });
        }
        else
        {
            RoundLog roundLog = GetCurrentRound();
            if (roundLog != null)
            {
                EventLog eventLog = new()
                {
                    logged_at = BattleManager.Instance.ElapsedTime.ToString(),
                    actor = actor.ToString(),
                    target = target.ToString() ?? null,
                    data = data,
                };
                roundLog.state_events.Add(eventLog);
            }
        }
    }

    public static void LogPlayerEvents(
        PlayerSide actor,
        PlayerSide? target = null,
        float? startedAt = null,
        float? updatedAt = null,
        Dictionary<string, object> data = null)
    {
        RoundLog roundLog = GetCurrentRound();
        if (roundLog != null)
        {
            EventLog eventLog = new()
            {
                logged_at = BattleManager.Instance.ElapsedTime.ToString(),
                actor = actor.ToString(),
                target = target.ToString() ?? null,
                data = data,
            };

            eventLog.started_at = startedAt != null ? startedAt.ToString() : BattleManager.Instance.ElapsedTime.ToString();
            eventLog.updated_at = updatedAt != null ? updatedAt.ToString() : eventLog.updated_at = BattleManager.Instance.ElapsedTime.ToString();
            roundLog.action_events.Add(eventLog);
        }
    }
    private static RoundLog GetCurrentRound()
    {
        if (_battleLog.games[CurrentGameIndex].rounds.Count == 0)
        {
            Debug.LogWarning("No eventLog started yet.");
            return null;
        }

        return _battleLog.games[CurrentGameIndex].rounds[^1];
    }
    #endregion

    #region Saving methods
    public static void SaveCurrentGame()
    {
        var json = JsonConvert.SerializeObject(_battleLog.games[CurrentGameIndex], Formatting.Indented);
        var savePath = Path.Combine(_logFolderPath, $"game_{CurrentGameIndex}.json");
        File.WriteAllText(savePath, json);
    }

    public static void SaveBattle()
    {
        var json = JsonConvert.SerializeObject(_battleLog, Formatting.Indented);
        var savePath = Path.Combine(_logFolderPath, "metadata.json");
        File.WriteAllText(savePath, json);
    }

    public static void SortAndSave()
    {
        _battleLog.games[CurrentGameIndex].rounds.ForEach((rounds) =>
        {
            rounds.action_events = rounds.action_events.OrderBy(log => float.Parse(log?.started_at ?? "0")).ToList();
        });
        SaveCurrentGame();
    }
    #endregion
}
