using System.Collections.Generic;
using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Leaderboard boundary. A "leaderboardId" namespaces separate boards (e.g. "wins",
    /// "rating"). LocalLeaderboardService keeps boards in a JSON file so a single
    /// machine has a working leaderboard now; a UGS Leaderboards implementation can
    /// replace it to make it global, with the same method shapes.
    /// </summary>
    public interface ILeaderboardService
    {
        /// <summary>
        /// Record a score for the current player on a board. Whether a higher or lower
        /// score wins, and whether it overwrites or accumulates, is the implementation's
        /// policy (default: keep the player's best).
        /// </summary>
        Task<ServiceResult> SubmitScoreAsync(string leaderboardId, double score);

        /// <summary>Get the top N entries for a board, already ranked.</summary>
        Task<ServiceResult<List<LeaderboardEntry>>> GetTopScoresAsync(string leaderboardId, int count = 20);

        /// <summary>Get the current player's own entry (with rank), or null if unranked.</summary>
        Task<ServiceResult<LeaderboardEntry>> GetPlayerEntryAsync(string leaderboardId);
    }
}
