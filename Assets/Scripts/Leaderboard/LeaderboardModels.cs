using System;
using System.Collections.Generic;
using SumoInput;
using SumoManager;

namespace SumoLeaderboard
{
    /// <summary>
    /// Top-level game mode dimension. Only Multiplayer produces data today;
    /// Campaign is reserved for when campaign mode ships (its board shows the
    /// empty state until then).
    /// </summary>
    public enum GameMode
    {
        Multiplayer = 0,
        Campaign = 1,
    }

    /// <summary>
    /// First leaderboard filter dimension (see Wiki leaderboard mockup dropdown 1).
    /// </summary>
    public enum PlayerMode
    {
        PvP = 0,
        PvAI = 1,
        AIvAI = 2,
    }

    /// <summary>
    /// Second leaderboard filter dimension (dropdown 2).
    /// Order and values MUST stay aligned with the DropdownControlModes options
    /// in MainMenu.unity: Buttons / Live Commands / Visual Script / AI Script.
    /// </summary>
    public enum ControlCategory
    {
        Buttons = 0,
        LiveCommands = 1,
        VisualScript = 2, // reserved: no InputType maps here yet
        AIScript = 3,
    }

    public static class LeaderboardMapping
    {
        public static ControlCategory ToControlCategory(this InputType type)
        {
            switch (type)
            {
                case InputType.LiveCommand:
                    return ControlCategory.LiveCommands;
                case InputType.Script:
                    return ControlCategory.AIScript;
                case InputType.UI:
                case InputType.Keyboard:
                default:
                    return ControlCategory.Buttons;
            }
        }

        /// <summary>
        /// A side is considered an AI when it is script-driven; UI, keyboard and
        /// live-command control are considered human (wiki §3.2).
        /// </summary>
        public static bool IsAI(this InputType type) => type == InputType.Script;

        public static PlayerMode DerivePlayerMode(InputType left, InputType right)
        {
            bool leftBot = left.IsAI();
            bool rightBot = right.IsAI();
            if (leftBot && rightBot) return PlayerMode.AIvAI;
            if (leftBot || rightBot) return PlayerMode.PvAI;
            return PlayerMode.PvP;
        }
    }

    [Serializable]
    public class LeaderboardEntry
    {
        public string ProfileID;    // persistent GUID, or "bot:<BotID>" for AI entries
        public string PlayerName;
        public string BotName;      // Bot.ID, or "-" for human non-script play
        public int Rating = EloCalculator.InitialRating;
        public int GamesPlayed;
        public int Wins;
        public int Losses;
        public int Draws;
        public int BestStreak;
        public int CurrentStreak;
        public string UpdatedAtUtc;
    }

    [Serializable]
    public class LeaderboardTable
    {
        public GameMode GameMode; // defaults to Multiplayer for v1 files (enum 0)
        public PlayerMode Mode;
        public ControlCategory Control;
        public List<LeaderboardEntry> Entries = new();
    }

    [Serializable]
    public class LeaderboardData
    {
        /// <summary>Bump when the schema changes; LeaderboardStore migrates older files.</summary>
        public const int CurrentVersion = 2;

        public int Version = CurrentVersion;
        public List<LeaderboardTable> Tables = new();
        public string Checksum;
    }

    /// <summary>
    /// What RecordBattle did to the ratings — used by the post-battle screen
    /// to show "+16 / -16" next