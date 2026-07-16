using System;

namespace SumoServices
{
    /// <summary>
    /// A single ranked row. Rank is assigned by the service at query time (1-based).
    /// Shape matches what UGS Leaderboards returns so the UI won't change when the
    /// backend does.
    /// </summary>
    [Serializable]
    public class LeaderboardEntry
    {
        public int Rank;
        public string PlayerId;
        public string DisplayName;
        public double Score;

        /// <summary>Unix seconds when this score was recorded. Used to break ties (earliest wins).</summary>
        public long Timestamp;
    }
}
