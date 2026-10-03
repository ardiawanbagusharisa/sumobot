using System;
using System.Threading.Tasks;
using SumoMultiplayer;
using UnityEngine;

namespace SumoServices
{
    /// <summary>
    /// Local, offline implementation of IAuthService. Generates a device-bound
    /// anonymous PlayerId on first run and reuses it forever, so leaderboard entries
    /// and saved data survive restarts even with no server. Swap this for a
    /// Photon/UGS-backed service later without touching any caller.
    /// </summary>
    public class LocalAuthService : IAuthService
    {
        private const string LegacyPlayerIdKey = "sumo.auth.playerId";
        private const string LegacyDisplayNameKey = "sumo.auth.displayName";
        private readonly string playerIdKey;
        private readonly string displayNameKey;
        private readonly bool migrateLegacyProfile;

        public LocalAuthService(string profileNamespace = null)
        {
            string profile = string.IsNullOrWhiteSpace(profileNamespace)
                ? OnlineBattleSession.ProfileName
                : profileNamespace.Trim();
            playerIdKey = $"sumo.auth.{profile}.playerId";
            displayNameKey = $"sumo.auth.{profile}.displayName";
            migrateLegacyProfile = string.Equals(profile, "sumobot_default", StringComparison.Ordinal);
        }

        public PlayerAccount Current { get; private set; }
        public bool IsSignedIn => Current != null && Current.IsValid;

        public Task<ServiceResult<PlayerAccount>> SignInAnonymouslyAsync()
        {
            string playerId = PlayerPrefs.GetString(playerIdKey, null);
            if (string.IsNullOrEmpty(playerId) && migrateLegacyProfile)
                playerId = PlayerPrefs.GetString(LegacyPlayerIdKey, null);
            if (string.IsNullOrEmpty(playerId))
            {
                playerId = "local_" + Guid.NewGuid().ToString("N");
                PlayerPrefs.SetString(playerIdKey, playerId);
                PlayerPrefs.Save();
            }

            string displayName = PlayerPrefs.GetString(
                displayNameKey,
                migrateLegacyProfile
                    ? PlayerPrefs.GetString(LegacyDisplayNameKey, "Player")
                    : "Player");

            PlayerPrefs.SetString(playerIdKey, playerId);
            PlayerPrefs.SetString(displayNameKey, displayName);
            PlayerPrefs.Save();

            Current = new PlayerAccount
            {
                PlayerId = playerId,
                DisplayName = displayName,
                IsGuest = true
            };

            Logger.Info($"[Auth] Signed in anonymously as {Current.DisplayName} ({Current.PlayerId})");
            return Task.FromResult(ServiceResult<PlayerAccount>.Ok(Current));
        }

        public Task<ServiceResult> SetDisplayNameAsync(string displayName)
        {
            if (!IsSignedIn)
                return Task.FromResult(ServiceResult.Fail("Not signed in."));

            if (string.IsNullOrWhiteSpace(displayName))
                return Task.FromResult(ServiceResult.Fail("Display name cannot be empty."));

            Current.DisplayName = displayName.Trim();
            PlayerPrefs.SetString(displayNameKey, Current.DisplayName);
            PlayerPrefs.Save();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> SignOutAsync()
        {
            Current = null;
            return Task.FromResult(ServiceResult.Ok());
        }
    }
}
