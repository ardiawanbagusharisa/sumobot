using System;
using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Persistence boundary for a player's save data (currency + inventory + equipped).
    /// LocalPlayerDataService writes JSON to disk today; a Cloud Save implementation
    /// can replace it later. Callers get an in-memory Current after Load and mutate
    /// through the grant/spend helpers so persistence stays centralized.
    /// </summary>
    public interface IPlayerDataService
    {
        /// <summary>The loaded save for the signed-in player, or null before Load.</summary>
        PlayerData Current { get; }

        /// <summary>
        /// The loadout currently being edited (equip/unequip act on this one). Defaults to
        /// the Left side's loadout after Load; change it with <see cref="SelectLoadout"/>.
        /// Null before Load. See decision-5.
        /// </summary>
        Loadout ActiveLoadout { get; }

        /// <summary>
        /// Raised after the coin balance changes and persists (spend/earn) or a new save
        /// loads. Lets UI (e.g. CoinBalanceView) refresh without callers pushing updates.
        /// </summary>
        event Action CoinsChanged;

        /// <summary>
        /// Raised after the owned-item set changes and persists (grant/revoke) or a new
        /// save loads. Lets the Inventory UI rebuild itself instead of being told to.
        /// </summary>
        event Action InventoryChanged;

        /// <summary>
        /// Raised after the equipped-costume set changes and persists (equip), or a new save
        /// loads. Lets the equip UI / bot preview refresh without callers pushing updates.
        /// </summary>
        event Action EquipmentChanged;

        /// <summary>Load (or create) the save for the given player id.</summary>
        Task<ServiceResult<PlayerData>> LoadAsync(string playerId);

        /// <summary>Persist the current save.</summary>
        Task<ServiceResult> SaveAsync();

        /// <summary>
        /// Wipe the current player's save back to a fresh default and persist it. Fires
        /// CoinsChanged / InventoryChanged / EquipmentChanged. Does not re-grant default items —
        /// GameServices.ResetPlayerProgressAsync layers that on top.
        /// </summary>
        Task<ServiceResult> ResetAsync();

        /// <summary>Add an item to the inventory and persist. No-op if already owned.</summary>
        Task<ServiceResult> GrantItemAsync(string itemId);

        /// <summary>Remove an item from the inventory (and unequip it) and persist. No-op if not owned.</summary>
        Task<ServiceResult> RevokeItemAsync(string itemId);

        /// <summary>Add coins to the balance and persist. Amount must be non-negative.</summary>
        Task<ServiceResult> AddCoinsAsync(int amount);

        /// <summary>Deduct coins and persist. Fails without spending if the balance is insufficient.</summary>
        Task<ServiceResult> TrySpendCoinsAsync(int amount);

        /// <summary>
        /// Choose which loadout <see cref="EquipAsync"/>/<see cref="UnequipAsync"/> act on and
        /// which <see cref="ActiveLoadout"/> returns. Fires EquipmentChanged when the selection
        /// changes so live views (preview, tabs, equipped marks) refresh through their existing
        /// subscription. No-op if the id is unknown or already active.
        /// </summary>
        void SelectLoadout(string loadoutId);

        /// <summary>Equip an owned item into a slot of the active loadout and persist. Fails if not owned.</summary>
        Task<ServiceResult> EquipAsync(string slot, string itemId);

        /// <summary>Clear a slot's equipped item on the active loadout and persist. No-op if already empty.</summary>
        Task<ServiceResult> UnequipAsync(string slot);

        /// <summary>
        /// Add coins to a specific player's save by id and persist it, without touching
        /// Current. Local-only convenience for crediting a Market seller who isn't the
        /// signed-in player; a networked implementation would do this server-side instead.
        /// Amount must be non-negative.
        /// </summary>
        Task<ServiceResult> AddCoinsToPlayerAsync(string playerId, int amount);
    }
}
