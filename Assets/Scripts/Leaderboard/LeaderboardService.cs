using System;
using System.Collections.Generic;
using System.Linq;
using SumoManager;
using UnityEngine;

namespace SumoLeaderboard
{
    /// <summary>
    /// Runtime facade over the leaderboard data. Lazy singleton (same pattern
    /// as GameManager) so no scene needs to pre-place it.
    /// </summary>
    public class LeaderboardService : MonoBehaviour, ILeaderboardService
    {
        private static LeaderboardService instance;

        /// <summary>True when the singleton already exists (avoids creating one during teardown).</summary>
        public static bool HasInstance => instance != null;

        public static LeaderboardService Instance
        {
            get
            {
                if (instance == null)
                {
                    GameObject go = new("LeaderboardService");
                    instance = go.AddComponent<LeaderboardService>();
                    DontDestroyOnLoad(go);
                }
                return instance;
            }
        }

        /// <summary>Raised after a battle result has been recorded and saved.</summary>
        public event Action Updated;

        /// <summary>
        /// Optional hook for external systems (e.g. a login/cloud-save layer):
        /// set this before the service is first touched and it will be used
        /// instead of the default local-file store.
        /// </summary>
        public static Func<ILeaderboardStore> StoreFactory;

        private ILeaderboardStore store;
        private LeaderboardData data;

        void Awake()
        {
            if (instance != null && instance != this)
            {
                Destroy(gameObject);
                return;
            }
            instance = this;
            DontDestroyOnLoad(gameObject);

            store = StoreFactory?.Invoke() ?? new LeaderboardStore();
            data = store.Load();
        }

        /// <summary>
        /// Swaps the persistence backend at runtime (e.g. right after login,
        /// to a store that syncs with the account server). Reloads data from
        /// the new store by default.
        /// </summary>
        public void SetStore(ILeaderboardStore newStore, bool reload = true)
        {
            if (newStore == null)
                return;
            store = newStore;
            if (reload)
            {
                data = store.Load();
                Updated?.Invoke();
            }
        }

        #region Query API

        /// <summary>Entries of one board, sorted by rank (copy — safe to mutate).</summary>
        public List<LeaderboardEntry> GetTable(GameMode gameMode, PlayerMode mode, ControlCategory control)
        {
            LeaderboardTable table = FindTable(gameMode, mode, control);
            if (table == null)
                return new List<LeaderboardEntry>();

            List<LeaderboardEntry> sorted = new(table.Entries);
            sorted.Sort(CompareEntries);
            return sorted;
        }

        /// <summary>1-based rank of a profile on a board, or -1 when absent.</summary>
        public int GetRank(string profileId, GameMode gameMode, PlayerMode mode, ControlCategory control)
        {
            List<LeaderboardEntry> sorted = GetTable(gameMode, mode, control);
            for (int i = 0; i < sorted.Count; i++)
                if (sorted[i].ProfileID == profileId)
                    return i + 1;
            return -1;
        }

        public LeaderboardEntry GetEntry(string profileId, GameMode gameMode, PlayerMode mode, ControlCategory control)
        {
            return FindTable(gameMode, mode, control)?.Entries.FirstOrDefault(e => e.ProfileID == profileId);
        }

        #endregion

        #region Record API

        /// <summary>
        /// Applies one finished battle to the boards. Each side is scored on its
        /// own (mode × control) board; Elo is computed once from both sides'
        /// current ratings, so cross-board matches (e.g. Buttons vs AI Script in
        /// PvAI) stay consistent.
        /// </summary>
        public LeaderboardOutcome? RecordBattle(BattleResultRecord record)
        {
            if (string.IsNullOrEmpty(record.LeftProfileID) || string.IsNullOrEmpty(record.RightProfileID))
            {
                Logger.Error("[LeaderboardService] Missing profile IDs, result not recorded.");
                return null;
            }

            // Self-play (same identity on both sides) is excluded to prevent farming.
            if (record.LeftProfileID == record.RightProfileID)
            {
                Logger.Info("[LeaderboardService] Self-play result ignored.");
                return null;
            }

            LeaderboardEntry left = GetOrCreateEntry(
                record.GameMode, record.Mode, record.LeftControl, record.LeftProfileID, record.LeftName, record.LeftBot);
            LeaderboardEntry right = GetOrCreateEntry(
                record.GameMode, record.Mode, record.RightControl, record.RightProfileID, record.RightName, record.RightBot);

            float leftScore = record.Winner switch
            {
                BattleWinner.Left => EloCalculator.Win,
                BattleWinner.Right => EloCalculator.Loss,
                _ => EloCalculator.Draw,
            };

            int oldLeft = left.Rating;
            int oldRight = right.Rating;
            (int newLeft, int newRight) = EloCalculator.UpdatePair(left.Rating, right.Rating, leftScore);
            left.Rating = newLeft;
            right.Rating = newRight;

            ApplyOutcome(left, leftScore, record.LeftName, record.LeftBot);
            ApplyOutcome(right, 1f - leftScore, record.RightName, record.RightBot);

            store.Save(data);
            Logger.Info($"[LeaderboardService] Recorded battle: {record.LeftName} ({newLeft}) vs {record.RightName} ({newRight}), winner={record.Winner}");
            Updated?.Invoke();

            return new LeaderboardOutcome
            {
                LeftRating = newLeft,
                RightRating = newRight,
                LeftDelta = newLeft - oldLeft,
                RightDelta = newRight - oldRight,
            };
        }

        /// <summary>
        /// Moves every board entry from one profile ID to another. Meant for
        /// login systems: when an anonymous local player signs in, call this
        /// with the old device GUID and the new account ID so their ratings
        /// carry over. If the account already has an entry on a board, that
        /// entry wins and the anonymous duplicate is dropped.
        /// </summary>
        public void ReassignProfile(string oldProfileId, string newProfileId, string newName = null)
        {
            if (string.IsNullOrEmpty(oldProfileId) || string.IsNullOrEmpty(newProfileId))
                return;
            if (oldProfileId == newProfileId)
                return;
            if (newProfileId.StartsWith("bot:"))
            {
                Logger.Error("[LeaderboardService] Account IDs must not use the reserved 'bot:' prefix.");
                return;
            }

            bool changed = false;
            foreach (LeaderboardTable table in data.Tables)
            {
                LeaderboardEntry old = table.Entries.FirstOrDefault(e => e.ProfileID == oldProfileId);
                if (old == null)
                    continue;

                LeaderboardEntry existing = table.Entries.FirstOrDefault(e => e.ProfileID == newProfileId);
                if (existing != null)
                {
                    // The account already played on this board; keep its record.
                    table.Entries.Remove(old);
                    Logger.Info($"[LeaderboardService] Dropped anonymous duplicate on {table.GameMode}/{table.Mode}/{table.Control}.");
                }
                else
                {
                    old.ProfileID = newProfileId;
                    if (!string.IsNullOrWhiteSpace(newName))
                        old.PlayerName = newName;
                }
                changed = true;
            }

            if (changed)
            {
                store.Save(data);
                Updated?.Invoke();
            }
        }

        /// <summary>
        /// Renames a profile on every board it appears on (display name only;
        /// ProfileID stays the ranking key). Saves when anything changed.
        /// </summary>
        public void RenameProfile(string profileId, string newName)
        {
            if (string.IsNullOrWhiteSpace(newName) || string.IsNullOrEmpty(profileId))
                return;

            bool changed = false;
            foreach (LeaderboardTable table in data.Tables)
                foreach (LeaderboardEntry entry in table.Entries)
                    if (entry.ProfileID == profileId && entry.PlayerName != newName)
                    {
                        entry.PlayerName = newName;
                        changed = true;
                    }

            if (changed)
            {
                store.Save(data);
                Updated?.Invoke();
            }
        }

        #endregion

        #region Internals

        private static void ApplyOutcome(LeaderboardEntry entry, float score, string name, string bot)
        {
            entry.GamesPlayed++;
            if (score >= EloCalculator.Win)
            {
                entry.Wins++;
                entry.CurrentStreak++;
                entry.BestStreak = Math.Max(entry.BestStreak, entry.CurrentStreak);
            }
            else if (score <= EloCalculator.Loss)
            {
                entry.Losses++;
                entry.CurrentStreak = 0;
            }
            else
            {
                entry.Draws++;
                entry.CurrentStreak = 0;
            }

            // Refresh display fields (player may have been renamed).
            if (!string.IsNullOrEmpty(name)) entry.PlayerName = name;
            if (!string.IsNullOrEmpty(bot)) entry.BotName = bot;
            entry.UpdatedAtUtc = DateTime.UtcNow.ToString("o");
        }

        private LeaderboardTable FindTable(GameMode gameMode, PlayerMode mode, ControlCategory control)
        {
            return data.Tables.FirstOrDefault(t => t.GameMode == gameMode && t.Mode == mode && t.Control == control);
        }

        private LeaderboardEntry GetOrCreateEntry(
            GameMode gameMode, PlayerMode mode, ControlCategory control, string profileId, string name, string bot)
        {
            LeaderboardTable table = FindTable(gameMode, mode, control);
            if (table == null)
            {
                table = new LeaderboardTable { GameMode = gameMode, Mode = mode, Control = control };
                data.Tables.Add(table);
            }

            LeaderboardEntry entry = table.Entries.FirstOrDefault(e => e.ProfileID == profileId);
            if (entry == null)
            {
                entry = new LeaderboardEntry
                {
                    ProfileID = profileId,
                    PlayerName = name,
                    BotName = string.IsNullOrEmpty(bot) ? "-" : bot,
                };
                table.Entries.Add(entry);
            }
            return entry;
        }

        /// <summary>Deterministic ranking: rating desc, wins desc, fewer games first, then name.</summary>
        public static int CompareEntries(LeaderboardEntry a, LeaderboardEntry b)
        {
            int byRating = b.Rating.CompareTo(a.Rating);
            if (byRating != 0) return byRating;
            int byWins = b.Wins.CompareTo(a.Wins);
            if (byWins != 0) return byWins;
            int byGames = a.GamesPlayed.CompareTo(b.GamesPlayed);
            if (byGames != 0) return byGames;
            return string.Compare(a.PlayerName, b.PlayerName, StringComparison.OrdinalIgnoreCase);
        }

        #endregion
    }
}
