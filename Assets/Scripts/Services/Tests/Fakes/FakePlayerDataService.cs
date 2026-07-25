using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace SumoServices.Tests
{
    /// <summary>
    /// In-memory IPlayerDataService for unit tests. Holds a save per player id so a test can
    /// switch the acting identity (LoadAsync) and credit other players (AddCoinsToPlayerAsync) —
    /// which is exactly what the Market buy flow exercises (buyer != seller on one machine).
    /// Money/inventory guards mirror LocalPlayerDataService; equip/loadout methods are
    /// simplified to just enough to satisfy the interface.
    /// </summary>
    public class FakePlayerDataService : IPlayerDataService
    {
        private readonly Dictionary<string, PlayerData> saves = new();

        public PlayerData Current { get; private set; }

        public Loadout ActiveLoadout =>
            Current != null && Current.Loadouts.Count > 0 ? Current.Loadouts[0] : null;

        public event Action CoinsChanged;
        public event Action InventoryChanged;
        public event Action EquipmentChanged;

        /// <summary>Test helper: read any player's save (e.g. to assert a seller was credited).</summary>
        public PlayerData SaveFor(string playerId) => saves.TryGetValue(playerId, out var d) ? d : null;

        private PlayerData GetOrCreate(string playerId)
        {
            if (!saves.TryGetValue(playerId, out var data))
            {
                data = PlayerData.CreateDefault(playerId);
                saves[playerId] = data;
            }
            return data;
        }

        public Task<ServiceResult<PlayerData>> LoadAsync(string playerId)
        {
            if (string.IsNullOrEmpty(playerId))
                return Task.FromResult(ServiceResult<PlayerData>.Fail("playerId is required."));

            Current = GetOrCreate(playerId);
            CoinsChanged?.Invoke();
            InventoryChanged?.Invoke();
            EquipmentChanged?.Invoke();
            return Task.FromResult(ServiceResult<PlayerData>.Ok(Current));
        }

        public Task<ServiceResult> SaveAsync() => Task.FromResult(ServiceResult.Ok());

        public Task<ServiceResult> ResetAsync()
        {
            if (Current == null) return Task.FromResult(ServiceResult.Fail("Nothing loaded."));

            var fresh = PlayerData.CreateDefault(Current.PlayerId);
            saves[Current.PlayerId] = fresh;
            Current = fresh;
            CoinsChanged?.Invoke();
            InventoryChanged?.Invoke();
            EquipmentChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> GrantItemAsync(string itemId)
        {
            if (Current == null) return Task.FromResult(ServiceResult.Fail("Nothing loaded."));
            if (string.IsNullOrEmpty(itemId)) return Task.FromResult(ServiceResult.Fail("itemId is required."));

            if (Current.OwnedItemIds.Contains(itemId)) return Task.FromResult(ServiceResult.Ok());
            Current.OwnedItemIds.Add(itemId);
            InventoryChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> RevokeItemAsync(string itemId)
        {
            if (Current == null) return Task.FromResult(ServiceResult.Fail("Nothing loaded."));
            if (Current.OwnedItemIds.Remove(itemId)) InventoryChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> AddCoinsAsync(int amount)
        {
            if (Current == null) return Task.FromResult(ServiceResult.Fail("Nothing loaded."));
            if (amount < 0) return Task.FromResult(ServiceResult.Fail("amount must be non-negative."));
            if (amount == 0) return Task.FromResult(ServiceResult.Ok());

            Current.Coins += amount;
            CoinsChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> TrySpendCoinsAsync(int amount)
        {
            if (Current == null) return Task.FromResult(ServiceResult.Fail("Nothing loaded."));
            if (amount < 0) return Task.FromResult(ServiceResult.Fail("amount must be non-negative."));
            if (Current.Coins < amount) return Task.FromResult(ServiceResult.Fail("Insufficient coins."));
            if (amount == 0) return Task.FromResult(ServiceResult.Ok());

            Current.Coins -= amount;
            CoinsChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }

        public void SelectLoadout(string loadoutId) => EquipmentChanged?.Invoke();

        public Task<ServiceResult> EquipAsync(string slot, string itemId)
        {
            if (Current == null) return Task.FromResult(ServiceResult.Fail("Nothing loaded."));
            if (!Current.Owns(itemId)) return Task.FromResult(ServiceResult.Fail("Cannot equip an item the player does not own."));

            var loadout = ActiveLoadout;
            if (loadout == null) return Task.FromResult(ServiceResult.Fail("No active loadout."));
            loadout.EquippedBySlot[slot] = itemId;
            EquipmentChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> UnequipAsync(string slot)
        {
            if (Current == null) return Task.FromResult(ServiceResult.Fail("Nothing loaded."));
            var loadout = ActiveLoadout;
            if (loadout != null && loadout.EquippedBySlot.Remove(slot)) EquipmentChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }

        public Task<ServiceResult> AddCoinsToPlayerAsync(string playerId, int amount)
        {
            if (string.IsNullOrEmpty(playerId)) return Task.FromResult(ServiceResult.Fail("playerId is required."));
            if (amount < 0) return Task.FromResult(ServiceResult.Fail("amount must be non-negative."));
            if (amount == 0) return Task.FromResult(ServiceResult.Ok());

            var target = GetOrCreate(playerId);
            target.Coins += amount;
            if (Current != null && Current.PlayerId == playerId) CoinsChanged?.Invoke();
            return Task.FromResult(ServiceResult.Ok());
        }
    }
}
