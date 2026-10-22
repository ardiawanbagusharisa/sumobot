using System;
using System.Collections.Generic;

namespace SumoLeaderboard
{
    /// <summary>
    /// Leaderboard boundary exposed through the composition root (GameServices.Leaderboard).
    /// LeaderboardService is the only implementation today (a MonoBehaviour singleton);
    /// this interface exists so GameServices can hold and hand out a reference to "the
    /// leaderboard" the same way it does for Auth/PlayerData/Catalog.
    /// </summary>
    public interface ILeaderboardService
    {
        /// <summary>Raised after a battle result has been recorded and saved.</summary>
        event Action Updated;

        /// <summary>Swaps the persistence backend at runtime (e.g. right after login, to a store that syncs with the account server).</summary>
        void SetStore(ILeaderboardStore newStore, bool reload = true);

        /// <summary>Entries of one board, sorted by rank (copy — safe to mutate).</summary>
        List<LeaderboardEntry> GetTable(GameMode gameMode, PlayerMode mode, ControlCategory control);

        /// <summary>1-based rank of a profile on a board, or -1 when absent.</summary>
        int GetRank(string profileId, GameMode gameMode, PlayerMode mode, ControlCategory control);

        /// <summary>The profile's own entry on a board, or null when absent.</summary>
        LeaderboardEntry GetEntry(string profileId, GameMode gameMode, PlayerMode mode, ControlCategory control);

        /// <summary>Applies one finished battle to the boards; returns the Elo delta, or null when the result wasn't recorded (missing IDs, self-play).</summary>
        LeaderboardOutcome? RecordBattle(BattleResultRecord record);

        /// <summary>Moves every board entry from one profile ID to another (e.g. anonymous device GUID to a signed-in account ID).</summary>
        void ReassignProfile(string oldProfileId, string newProfileId, string newName = null);

        /// <summary>Renames a profile's display name on every board it appears on.</summary>
        void RenameProfile(string profileId, string newName);
    }
}
