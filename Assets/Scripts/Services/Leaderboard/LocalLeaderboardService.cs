using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

namespace SumoServices
{
    /// <summary>
    /// Local JSON-file leaderboard. Boards live in one file under
    /// Application.persistentDataPath. It keeps each player's BEST score per board and
    /// ranks by score descending (ties broken by earliest timestamp). Depends on
    /// IAuthService only to know who is submitting — same as the cloud version would.
    /// </summary>
    public class LocalLeaderboardService : ILeaderboardService
    {
        private readonly IAuthService auth;

        // leaderboardId -> (playerId -> entry)
        private Dictionary<string, Dictionary<string, LeaderboardEntry>> boards;

        private static string FilePath => Path.Combine(Application.persistentDataPath, "leaderboards.json");

        public LocalLeaderboardService(IAuthService auth)
        {
            this.auth = auth;
        }

        private void EnsureLoaded()
        {
            if (boards != null) return;

            try
            {
                if (File.Exists(FilePath))
                {
                    string json = File.ReadAllText(FilePath);
                    boards = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, LeaderboardEntry>>>(json);
                }
            }
            catch (Exception e)
            {
                Logger.Error($"[Leaderboard] Load failed, starting empty: {e.Message}");
            }

            boards ??= new();
        }

        private void Persist()
        {
            try
            {
                string json = JsonConvert.SerializeObject(boards, Formatting.Indented);
                File.WriteAllText(FilePath, json);
            }
            catch (Exception e)
            {
                Logger.Error($"[Leaderboard] Save failed: {e.Message}");
            }
        }

        public Task<ServiceResult> SubmitScoreAsync(string leaderboardId, double score)
        {
            if (auth?.Current == null || !auth.Current.IsValid)
                return Task.FromResult(ServiceResult.Fail("Must be signed in to submit a score."));
            if (string.IsNullOrEmpty(leaderboardId))
                return Task.FromResult(ServiceResult.Fail("leaderboardId is required."));

            EnsureLoaded();

            if (!boards.TryGetValue(leaderboardId, out var board))
            {
                board = new Dictionary<string, LeaderboardEntry>();
                boards[leaderboardId] = board;
            }

            string playerId = auth.Current.PlayerId;
            bool hasExisting = board.TryGetValue(playerId, out var existing);

            // Keep the player's best score only.
            if (!hasExisting || score > existing.Score)
            {
                board[playerId] = new LeaderboardEntry
                {
                    PlayerId = playerId,
                    DisplayName = auth.Current.DisplayName,
                    Score = score,
                    Timestamp = DateTimeOffset.UtcNow.ToUnixTimeSeconds()
                };
                Persist();
            }
            else if (existing.DisplayName != auth.Current.DisplayName)
            {
                // Keep the display name fresh even when the score isn't beaten.
                existing.DisplayName = auth.Current.DisplayName;
                Persist();
            }

            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult<List<LeaderboardEntry>>> GetTopScoresAsync(string leaderboardId, int count = 20)
        {
            EnsureLoaded();
            var ranked = RankedEntries(leaderboardId).Take(count).ToList();
            return Task.FromResult(ServiceResult<List<LeaderboardEntry>>.Ok(ranked));
        }

        public Task<ServiceResult<LeaderboardEntry>> GetPlayerEntryAsync(string leaderboardId)
        {
            if (auth?.Current == null || !auth.Current.IsValid)
                return Task.FromResult(ServiceResult<LeaderboardEntry>.Fail("Not signed in."));

            EnsureLoaded();
            var entry = RankedEntries(leaderboardId)
                .FirstOrDefault(e => e.PlayerId == auth.Current.PlayerId);
            return Task.FromResult(ServiceResult<LeaderboardEntry>.Ok(entry));
        }

        // Ranks a board's entries (score desc, then earliest timestamp) and stamps Rank.
        private List<LeaderboardEntry> RankedEntries(string leaderboardId)
        {
            if (boards == null || !boards.TryGetValue(leaderboardId, out var board))
                return new List<LeaderboardEntry>();

            var ordered = board.Values
                .OrderByDescending(e => e.Score)
                .ThenBy(e => e.Timestamp)
                .ToList();

            for (int i = 0; i < ordered.Count; i++)
                ordered[i].Rank = i + 1;

            return ordered;
        }
    }
}
