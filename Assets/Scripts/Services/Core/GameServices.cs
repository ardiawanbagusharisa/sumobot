using System.Threading.Tasks;
using SumoLeaderboard;

namespace SumoServices
{
    /// <summary>
    /// Single access point and composition root for all backend services. This is the
    /// ONE place that decides which concrete implementation each service uses, so
    /// swapping the local stubs for Photon / UGS later touches only this file.
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

        /// <summary>Shop buy, composed over Catalog (prices) and PlayerData (balance + inventory).</summary>
        public static ITradeService Trade { get; private set; }

        /// <summary>Player-to-player Market (Community tab): browse/list/unlist/reprice/buy listings.</summary>
        public static IMarketService Market { get; private set; }

        /// <summary>
        /// Backed by SumoLeaderboard.LeaderboardService.Instance (a MonoBehaviour
        /// singleton owned by the Leaderboard feature, not constructed here). Exposed
        /// through GameServices so callers reach it the same way as the other services.
        /// </summary>
        public static ILeaderboardService Leaderboard { get; private set; }

        public static bool IsInitialized { get; private set; }

        /// <summary>
        /// Wire up the concrete services. Call once. To move to cloud backends later,
        /// replace the constructors here (e.g. new UgsAuthService()) — nothing else changes.
        /// </summary>
        public static void Initialize()
        {
            if (IsInitialized) return;

            var auth = new LocalAuthService();
            Auth = auth;
            PlayerData = new LocalPlayerDataService();
            Catalog = new LocalCatalogService();
            Trade = new LocalTradeService(Catalog, PlayerData);
            Market = new LocalMarketService(Catalog, PlayerData);
            Leaderboard = LeaderboardService.Instance;

            IsInitialized = true;
        }

        /// <summary>
        /// Bootstrap a play session: sign in anonymously, load the catalog, load the
        /// player's save, and grant any free/default items they don't own yet. Returns
        /// the signed-in account on success. This is the single call the UI needs to
        /// go from "cold start" to "ready to play".
        /// </summary>
        public static async Task<ServiceResult<PlayerAccount>> StartSessionAsync()
        {
            if (!IsInitialized) Initialize();

            var signIn = await Auth.SignInAnonymouslyAsync();
            if (!signIn.Success)
                return ServiceResult<PlayerAccount>.Fail(signIn.Error);

            var catalog = await Catalog.LoadAsync();
            if (!catalog.Success)
                return ServiceResult<PlayerAccount>.Fail(catalog.Error);

            var load = await PlayerData.LoadAsync(signIn.Value.PlayerId);
            if (!load.Success)
                return ServiceResult<PlayerAccount>.Fail(load.Error);

            var market = await Market.LoadAsync();
            if (!market.Success)
                return ServiceResult<PlayerAccount>.Fail(market.Error);

            await GrantDefaultItemsAsync();

            return ServiceResult<PlayerAccount>.Ok(signIn.Value);
        }

        // Ensures the player owns every default catalog item (the parts the game ships
        // with). No purchase/currency flow exists yet — add one here only when the PM
        // defines how non-default items are acquired.
        private static async Task GrantDefaultItemsAsync()
        {
            foreach (var item in Catalog.AllItems)
            {
                if (item.IsDefault && !PlayerData.Current.Owns(item.Id))
                    await PlayerData.GrantItemAsync(item.Id);
            }
        }

        /// <summary>Test/reset hook: forget the wired services so Initialize runs fresh.</summary>
        public static void Reset()
        {
            Auth = null;
            PlayerData = null;
            Catalog = null;
            Trade = null;
            Market = null;
            Leaderboard = null;
            IsInitialized = false;
        }
    }
}
