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
    public static Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>> ActionLoggers = new Dictionary<PlayerSide, Dictionary<string, DebouncedLogger>>();

    private static BattleLog _battleLog;
    private static string _logFolderPath;

    public static int CurrentGameIndex => _battleLog.games.Count > 0 ? _battleLog.games[^1].index : 0;

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
        public string category;
        public bool isStart;

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

        // Bounce made by this player
        public int contact_made = 0;
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
                {
                    actionLogger.ForceStopAndSave();
                }
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

    #region Core Battle Log

    public static void InitLog()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        var folderName = $"battle_{timestamp}";

        _logFolderPath = Path.Combine(Application.persistentDataPath, "Logs", folderName);
        Directory.CreateDirectory(_logFolderPath);
    }

    public static void InitBattle()
    {
        _battleLog = new BattleLog();
        _battleLog.input_type = BattleManager.Instance.BattleInputType.ToString();
        _battleLog.battle_id = BattleManager.Instance.Battle.BattleID.ToString();
        _battleLog.countdown_time = BattleManager.Instance.CountdownTime;
        _battleLog.battle_time = BattleManager.Instance.BattleTime;
        _battleLog.round_type = (int)BattleManager.Instance.RoundSystem;
        SaveBattle();
    }

    public static void UpdateMetadata()
    {
        if (BattleManager.Instance.Bot.IsEnable)
        {
            var p1Bot = BattleManager.Instance.Bot.Left;
            var p2Bot = BattleManager.Instance.Bot.Right;

            if (p1Bot != null)
            {
                _battleLog.left_player_stats.bot = p1Bot?.ID ?? "";
            }

            if (p2Bot != null)
            {
                _battleLog.right_player_stats.bot = p2Bot?.ID ?? "";
            }
        }

        _battleLog.left_player_stats.skill_type = BattleManager.Instance.Battle.LeftPlayer.Skill.Type.ToString();
        _battleLog.right_player_stats.skill_type = BattleManager.Instance.Battle.RightPlayer.Skill.Type.ToString();

        if (_battleLog.games.Count > 0)
        {
            var currRound = GetCurrentRound();
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

    public static void StartNewGame()
    {
        int newGameIndex = CurrentGameIndex;
        if (_battleLog.games.Count > 0)
        {
            newGameIndex += 1;
        }
        var newGame = new GameLog { index = newGameIndex };
        newGame.timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        _battleLog.games.Add(newGame);
    }

    public static void StartRound(int index)
    {
        var newRound = new RoundLog { index = index };
        newRound.timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        _battleLog.games[CurrentGameIndex].rounds.Add(newRound);
    }

    // [winner] is either Draw, Left, or Side
    public static void SetRoundWinner(string winner)
    {
        Debug.Log($"Setting round winner {winner}");
        var round = GetCurrentRound();
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
        {
            _battleLog.left_player_stats.win_per_game += 1;
        }
        else if (winner == BattleWinner.Right)
        {
            _battleLog.right_player_stats.win_per_game += 1;
        }
    }

    public static void LogBattleState(
        LogActorType actor,
        bool includeInCurrentRound = false,
        LogActorType? target = null,
        Dictionary<string, object> data = null)
    {
        if (!includeInCurrentRound)
        {
            // adding to events in Metadata
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
            var round = GetCurrentRound();
            if (round != null)
            {
                EventLog roundLog = new()
                {
                    logged_at = BattleManager.Instance.ElapsedTime.ToString(),
                    actor = actor.ToString(),
                    target = target.ToString() ?? null,
                    data = data,
                };
                round.state_events.Add(roundLog);
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
        var round = GetCurrentRound();
        if (round != null)
        {
            EventLog roundLog = new()
            {
                logged_at = BattleManager.Instance.ElapsedTime.ToString(),
                actor = actor.ToString(),
                target = target.ToString() ?? null,
                data = data,
                isStart = isStart,
                category = category,
            };

            if (startedAt != null)
            {
                roundLog.started_at = startedAt.ToString();
            }
            else
            {
                roundLog.started_at = BattleManager.Instance.ElapsedTime.ToString();
            }

            if (updatedAt != null)
            {
                roundLog.updated_at = updatedAt.ToString();
            }
            else
            {
                roundLog.updated_at = BattleManager.Instance.ElapsedTime.ToString();
            }

            round.action_events.Add(roundLog);
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

    public static void SaveCurrentGame()
    {

        var json = JsonConvert.SerializeObject(_battleLog.games[CurrentGameIndex], Formatting.Indented);
        string paddedIndex = CurrentGameIndex.ToString("D3"); // D3 = 3-digit padding
        var savePath = Path.Combine(_logFolderPath, $"game_{paddedIndex}.json");
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

}
