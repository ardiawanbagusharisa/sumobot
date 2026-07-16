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

        /// <summary>Load (or create) the save for the given player id.</summary>
        Task<ServiceResult<PlayerData>> LoadAsync(string playerId);

        /// <summary>Persist the current save.</summary>
        Task<ServiceResult> SaveAsync();

        /// <summary>Add an item to the inventory and persist. No-op if already owned.</summary>
        Task<ServiceResult> GrantItemAsync(string itemId);

        /// <summary>Equip an owned item into a slot and persist. Fails if not owned.</summary>
        Task<ServiceResult> EquipAsync(string slot, string itemId);
    }
}
