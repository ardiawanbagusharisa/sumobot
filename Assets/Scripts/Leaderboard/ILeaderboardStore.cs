namespace SumoLeaderboard
{
    /// <summary>
    /// Persistence contract for leaderboard data. The default implementation
    /// (LeaderboardStore) writes a local JSON file; a login/cloud-save system
    /// can provide its own implementation and hand it to LeaderboardService
    /// via SetStore() or the static StoreFactory hook.
    /// </summary>
    public interface ILeaderboardStore
    {
        LeaderboardData Load();
        void Save(LeaderboardData data);
    }
}
