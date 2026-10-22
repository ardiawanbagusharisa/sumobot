using System;

namespace SumoServices
{
    /// <summary>
    /// The authenticated identity returned by an IAuthService. This is the single
    /// "who is playing" record that Leaderboard / Inventory / Market all key off.
    /// Deliberately backend-agnostic: PlayerId could come from a local anonymous
    /// id today, or from Photon / UGS Authentication later — callers never care.
    /// </summary>
    [Serializable]
    public class PlayerAccount
    {
        /// <summary>Stable unique id that persists across sessions. Never changes once created.</summary>
        public string PlayerId;

        /// <summary>Display name shown in UI / leaderboards. May be edited by the player.</summary>
        public string DisplayName;

        /// <summary>
        /// True when this account has no real credentials yet (anonymous / device-bound).
        /// An anonymous account can later be "upgraded" by linking an email or provider
        /// without changing PlayerId, so nothing tied to the id is lost.
        /// </summary>
        public bool IsGuest;

        public bool IsValid => !string.IsNullOrEmpty(PlayerId);
    }
}
