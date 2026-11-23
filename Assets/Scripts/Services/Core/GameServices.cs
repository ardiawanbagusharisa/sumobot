using System.Threading.Tasks;
using SumoLeaderboard;
using SumoMultiplayer;
using UnityEngine;

namespace SumoServices
{
    public enum AccountLoginMode { Guest, SignIn, SignUp }

    /// <summary>
    /// Single access point and composition root for all backend services. This is the
    /// ONE place that decides which concrete implementation each service uses, so
    /// moving local player data to a cloud service later touches this boundary.
    ///
    /// Usage: GameManager calls Initialize() once at startup, then the rest of the game
    /// reads GameServices.Auth / PlayerData / Catalog / Leaderboard.
    /// </summary>
    public static class GameServices
    {
        /// <summary>
        /// Guarantees the services are wired before any scene's Awake/Start runs, no
        /// matter which scene the game (or the Editor) starts from. This removes the
        /// old ordering hazard where callers had to touch GameManager.Instance first
        /// to trigger initialization. Runs once per play session.
        /// </summary>
        [UnityEngine.RuntimeInitializeOnLoadMethod(UnityEngine.RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Bootstrap() => Initialize();

        public static IAuthService Auth { get; private set; }
        public static IPlayerDataService PlayerData { get; private set; }
        public static ICatalogService Catalog { get; private set; }

        /// <summary>Market buy/sell, composed over Catalog (prices) and PlayerData (balance + inventory).</summary>
        public static ITradeService Trade { get; private set; }

        /// <summary>
        /// Backed by SumoLeaderboard.LeaderboardService.Instance (a MonoBehaviour
        /// singleton owned by the Leaderboard feature, not constructed here). Exposed
        /// through GameServices so callers reach it the same way as the other services.
        /// </summary>
        public static ILeaderboardService Leaderboard { get; private set; }

        public static bool IsInitialized { get; private set; }

        /// <summary>
        /// Wire up the concrete services. Call once. To move to cloud backends later,
        /// replace the constructors here when a backend changes.
        /// </summary>
        public static void Initialize()
        {
            if (IsInitialized) return;

            var auth = new UgsAuthService();
            Auth = auth;
            PlayerData = new LocalPlayerDataService();
            Catalog = new LocalCatalogService();
            Trade = new LocalTradeService(Catalog, PlayerData);
            Leaderboard = LeaderboardService.Instance;

            IsInitialized = true;
        }

        /// <summary>
        /// Bootstrap a play session: authenticate, load the catalog, load the
        /// player's save, and grant any free/default items they don't own yet. Returns
        /// the signed-in account on success. This is the single call the UI needs to
        /// go from "cold start" to "ready to play".
        /// </summary>
        public static async Task<ServiceResult<PlayerAccount>> StartSessionAsync(
            AccountLoginMode mode = AccountLoginMode.Guest,
            string username = null,
            string password = null)
        {
            if (!IsInitialized) Initialize();

            ServiceResult<PlayerAccount> signIn = mode switch
            {
                AccountLoginMode.SignIn => await Auth.SignInAsync(username, password),
                AccountLoginMode.SignUp => await Auth.SignUpAsync(username, password),
                _ => await Auth.SignInAnonymouslyAsync()
            };
            if (!signIn.Success)
                return ServiceResult<PlayerAccount>.Fail(signIn.Error);

            var catalog = await Catalog.LoadAsync();
            if (!catalog.Success)
                return ServiceResult<PlayerAccount>.Fail(catalog.Error);

            // Existing dummy accounts used local_<guid> save names. Import once
            // into the first UGS account used by this local launch profile.
            string profile = OnlineBattleSession.ProfileName;
            string marker = $"sumo.auth.{profile}.legacyImported";
            if (!PlayerPrefs.HasKey(marker) && PlayerData is LocalPlayerDataService localData)
            {
                var legacy = await new LocalAuthService(profile).SignInAnonymouslyAsync();
                if (legacy.Success && localData.ImportLegacySave(legacy.Value.PlayerId, signIn.Value.PlayerId))
                {
                    PlayerPrefs.SetString(marker, signIn.Value.PlayerId);
                    PlayerPrefs.Save();
                }
            }

            var load = await PlayerData.LoadAsync(signIn.Value.PlayerId);
            if (!load.Success)
                return ServiceResult<PlayerAccount>.Fail(load.Error);

            await GrantDefaultItemsAsync();

            return ServiceResult<PlayerAccount>.Ok(signIn.Value);
        }

        public static async Task<ServiceResult> SignOutAsync()
        {
            if (!IsInitialized) return ServiceResult.Ok();
            ServiceResult result = await Auth.SignOutAsync();
            if (result.Success && PlayerData is LocalPlayerDataService localData)
                localData.Unload();
            return result;
        }

        // Ensures the player owns every default catalog item shipped with the game.
        private static async Task GrantDefaultItemsAsync()
        {
            foreach (var item in Catalog.AllItems)
            {
                if (item.IsDefault && !PlayerData.Current.Owns(item.Id))
                    await PlayerData.GrantItemAsync(item.Id);

                if (item.IsDefault && !string.IsNullOrEmpty(item.Slot) &&
                    !PlayerData.Current.EquippedBySlot.ContainsKey(item.Slot))
                {
                    await PlayerData.EquipAsync(item.Slot, item.Id);
                }
            }
        }

        /// <summary>Test/reset hook: forget the wired services so Initialize runs fresh.</summary>
        public static void Reset()
        {
            Auth = null;
            PlayerData = null;
            Catalog = null;
            Trade = null;
            Leaderboard = null;
            IsInitialized = false;
        }
    }
}
