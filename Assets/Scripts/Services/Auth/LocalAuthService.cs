using System;
using System.Threading.Tasks;
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
        private const string PlayerIdKey = "sumo.auth.playerId";
        private const string DisplayNameKey = "sumo.auth.displayName";

        public PlayerAccount Current { get; private set; }
        public bool IsSignedIn => Current != null && Current.IsValid;

        public Task<ServiceResult<PlayerAccount>> SignInAnonymouslyAsync()
        {
            string playerId = PlayerPrefs.GetString(PlayerIdKey, null);
            if (string.IsNullOrEmpty(playerId))
            {
                playerId = "local_" + Guid.NewGuid().ToString("N");
                PlayerPrefs.SetString(PlayerIdKey, playerId);
                PlayerPrefs.Save();
            }

            string displayName = PlayerPrefs.GetString(DisplayNameKey, "Player");

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
            PlayerPrefs.SetString(DisplayNameKey, Current.DisplayName);
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
