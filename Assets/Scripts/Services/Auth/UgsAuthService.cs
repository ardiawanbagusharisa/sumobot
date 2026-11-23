using System;
using System.Threading.Tasks;
using SumoMultiplayer;
using Unity.Services.Authentication;
using Unity.Services.Core;
using Unity.Services.Core.Environments;
using UnityEngine;

namespace SumoServices
{
    /// <summary>One UGS identity for inventory, leaderboard, and online sessions.</summary>
    public sealed class UgsAuthService : IAuthService
    {
        private readonly string profile = OnlineBattleSession.ProfileName;
        private const string NameKeyPrefix = "sumo.ugs.displayName.";
        private const string LinkedKeyPrefix = "sumo.ugs.credentialLinked.";

        public PlayerAccount Current { get; private set; }
        public bool IsSignedIn => Current != null && AuthenticationService.Instance.IsSignedIn;

        private async Task InitializeAsync()
        {
            if (UnityServices.State == ServicesInitializationState.Uninitialized)
            {
                var options = new InitializationOptions();
                options.SetProfile(profile);
                string environment = OnlineLaunchOptions.GetEnvironmentName();
                if (!string.IsNullOrWhiteSpace(environment))
                    options.SetEnvironmentName(environment);
                await UnityServices.InitializeAsync(options);
            }

            if (AuthenticationService.Instance.Profile != profile)
                throw new InvalidOperationException(
                    $"UGS profile mismatch: expected '{profile}', got '{AuthenticationService.Instance.Profile}'.");
        }

        public async Task<ServiceResult<PlayerAccount>> SignInAnonymouslyAsync()
        {
            try
            {
                await InitializeAsync();
                if (!AuthenticationService.Instance.IsSignedIn)
                    await AuthenticationService.Instance.SignInAnonymouslyAsync();
                return ServiceResult<PlayerAccount>.Ok(CaptureAccount());
            }
            catch (Exception error)
            {
                return ServiceResult<PlayerAccount>.Fail(error.Message);
            }
        }

        public async Task<ServiceResult<PlayerAccount>> SignUpAsync(string username, string password)
        {
            if (string.IsNullOrWhiteSpace(username) || string.IsNullOrEmpty(password))
                return ServiceResult<PlayerAccount>.Fail("Enter a username and password.");
            try
            {
                await InitializeAsync();
                if (AuthenticationService.Instance.IsSignedIn)
                    AuthenticationService.Instance.SignOut(true);
                Current = null;
                await AuthenticationService.Instance.SignUpWithUsernamePasswordAsync(username.Trim(), password);
                return ServiceResult<PlayerAccount>.Ok(CaptureAccount(username.Trim(), true));
            }
            catch (Exception error)
            {
                return ServiceResult<PlayerAccount>.Fail(error.Message);
            }
        }

        public async Task<ServiceResult<PlayerAccount>> SignInAsync(string username, string password)
        {
            if (string.IsNullOrWhiteSpace(username) || string.IsNullOrEmpty(password))
                return ServiceResult<PlayerAccount>.Fail("Enter a username and password.");
            try
            {
                await InitializeAsync();
                if (AuthenticationService.Instance.IsSignedIn)
                    AuthenticationService.Instance.SignOut(true);
                Current = null;
                await AuthenticationService.Instance.SignInWithUsernamePasswordAsync(username.Trim(), password);
                return ServiceResult<PlayerAccount>.Ok(CaptureAccount(username.Trim(), true));
            }
            catch (Exception error)
            {
                return ServiceResult<PlayerAccount>.Fail(error.Message);
            }
        }

        public async Task<ServiceResult<PlayerAccount>> LinkUsernamePasswordAsync(string username, string password)
        {
            if (!IsSignedIn || Current == null || !Current.IsGuest)
                return ServiceResult<PlayerAccount>.Fail("Sign in as a guest before linking credentials.");
            if (string.IsNullOrWhiteSpace(username) || string.IsNullOrEmpty(password))
                return ServiceResult<PlayerAccount>.Fail("Enter a username and password.");
            try
            {
                await AuthenticationService.Instance.AddUsernamePasswordAsync(username.Trim(), password);
                Current.IsGuest = false;
                PlayerPrefs.SetInt(LinkedKeyPrefix + Current.PlayerId, 1);
                PlayerPrefs.Save();
                return ServiceResult<PlayerAccount>.Ok(Current);
            }
            catch (Exception error)
            {
                return ServiceResult<PlayerAccount>.Fail(error.Message);
            }
        }

        public Task<ServiceResult> SetDisplayNameAsync(string displayName)
        {
            if (!IsSignedIn)
                return Task.FromResult(ServiceResult.Fail("Not signed in."));
            if (string.IsNullOrWhiteSpace(displayName))
                return Task.FromResult(ServiceResult.Fail("Display name cannot be empty."));

            Current.DisplayName = displayName.Trim();
            PlayerPrefs.SetString(NameKeyPrefix + Current.PlayerId, Current.DisplayName);
            PlayerPrefs.Save();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> SignOutAsync()
        {
            if (UnityServices.State == ServicesInitializationState.Initialized &&
                AuthenticationService.Instance.IsSignedIn)
                AuthenticationService.Instance.SignOut(true);
            Current = null;
            return Task.FromResult(ServiceResult.Ok());
        }

        private PlayerAccount CaptureAccount(string fallbackName = null, bool credentialSignIn = false)
        {
            var auth = AuthenticationService.Instance;
            if (credentialSignIn)
            {
                PlayerPrefs.SetInt(LinkedKeyPrefix + auth.PlayerId, 1);
                PlayerPrefs.Save();
            }
            string username = auth.PlayerInfo?.Username;
            string defaultName = !string.IsNullOrWhiteSpace(username)
                ? username
                : !string.IsNullOrWhiteSpace(fallbackName) ? fallbackName : "Player";
            Current = new PlayerAccount
            {
                PlayerId = auth.PlayerId,
                DisplayName = PlayerPrefs.GetString(NameKeyPrefix + auth.PlayerId, defaultName),
                IsGuest = !credentialSignIn &&
                    !PlayerPrefs.HasKey(LinkedKeyPrefix + auth.PlayerId) &&
                    string.IsNullOrWhiteSpace(username)
            };
            return Current;
        }
    }
}
