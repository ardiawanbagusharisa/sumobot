using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

namespace SumoServices
{
    /// <summary>
    /// Local JSON-file implementation of IPlayerDataService. One file per player id
    /// under Application.persistentDataPath/PlayerData. This is the project's local
    /// "database" for now; because everything goes through this interface, moving to
    /// Cloud Save later means writing one new class and swapping it in GameServices.
    /// </summary>
    public class LocalPlayerDataService : IPlayerDataService
    {
        public PlayerData Current { get; private set; }

        public event Action CoinsChanged;
        public event Action InventoryChanged;
        public event Action EquipmentChanged;

        private readonly string dir;

        public LocalPlayerDataService(string rootPath = null)
        {
            dir = Path.Combine(rootPath ?? Application.persistentDataPath, "PlayerData");
        }

        public void Unload()
        {
            Current = null;
            CoinsChanged?.Invoke();
            InventoryChanged?.Invoke();
            EquipmentChanged?.Invoke();
        }

        /// <summary>Copies a legacy dummy save once; never replaces account data.</summary>
        public bool ImportLegacySave(string legacyPlayerId, string accountPlayerId)
        {
            if (string.IsNullOrWhiteSpace(legacyPlayerId) ||
                string.IsNullOrWhiteSpace(accountPlayerId) ||
                legacyPlayerId == accountPlayerId)
                return true;

            try
            {
                string source = PathFor(legacyPlayerId);
                string destination = PathFor(accountPlayerId);
                if (!File.Exists(source) || File.Exists(destination))
                    return true;

                PlayerData imported = JsonConvert.DeserializeObject<PlayerData>(File.ReadAllText(source));
                if (imported == null) return false;
                imported.PlayerId = accountPlayerId;
                Directory.CreateDirectory(dir);
                File.WriteAllText(destination, JsonConvert.SerializeObject(imported, Formatting.Indented));
                Logger.Info($"[PlayerData] Imported local save into Unity account {accountPlayerId}.");
                return true;
            }
            catch (Exception error)
            {
                Logger.Error($"[PlayerData] Legacy import failed: {error.Message}");
                return false;
            }
        }

        private string PathFor(string playerId) => Path.Combine(dir, $"{playerId}.json");

        public Task<ServiceResult<PlayerData>> LoadAsync(string playerId)
        {
            if (string.IsNullOrEmpty(playerId))
                return Task.FromResult(ServiceResult<PlayerData>.Fail("playerId is required."));

            try
            {
                string path = PathFor(playerId);
                if (File.Exists(path))
                {
                    string json = File.ReadAllText(path);
                    Current = JsonConvert.DeserializeObject<PlayerData>(json) ?? PlayerData.CreateDefault(playerId);
                }
                else
                {
                    Current = PlayerData.CreateDefault(playerId);
                }

                // Guard against partially-written / hand-edited files.
                Current.PlayerId = playerId;
                Current.OwnedItemIds ??= new();
                Current.EquippedBySlot ??= new();
                Current.CompletedCampaignLevels ??= new();

                // A freshly loaded save is a full state change: let any already-live view
                // (a coin counter, the inventory list) refresh to the loaded values.
                CoinsChanged?.Invoke();
                InventoryChanged?.Invoke();
                EquipmentChanged?.Invoke();

                return Task.FromResult(ServiceResult<PlayerData>.Ok(Current));
            }
            catch (Exception e)
            {
                Logger.Error($"[PlayerData] Load failed: {e.Message}");
                return Task.FromResult(ServiceResult<PlayerData>.Fail(e.Message));
            }
        }

        public Task<ServiceResult> SaveAsync()
        {
            if (Current == null)
                return Task.FromResult(ServiceResult.Fail("Nothing loaded to save."));

            try
            {
                Directory.CreateDirectory(dir);
                string json = JsonConvert.SerializeObject(Current, Formatting.Indented);
                File.WriteAllText(PathFor(Current.PlayerId), json);
                return Task.FromResult(ServiceResult.Ok());
            }
            catch (Exception e)
            {
                Logger.Error($"[PlayerData] Save failed: {e.Message}");
                return Task.FromResult(ServiceResult.Fail(e.Message));
            }
        }

        public async Task<ServiceResult> GrantItemAsync(string itemId)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (string.IsNullOrEmpty(itemId)) return ServiceResult.Fail("itemId is required.");

            if (Current.OwnedItemIds.Contains(itemId))
                return ServiceResult.Ok(); // already owned — no change, no event

            Current.OwnedItemIds.Add(itemId);

            var save = await SaveAsync();
            if (save.Success) InventoryChanged?.Invoke();
            return save;
        }

        public async Task<ServiceResult> RevokeItemAsync(string itemId)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (string.IsNullOrEmpty(itemId)) return ServiceResult.Fail("itemId is required.");

            bool removed = Current.OwnedItemIds.Remove(itemId);
            bool unequipped = false;

            // Drop any equip slot that pointed at the now-unowned item so the save stays consistent.
            var slots = new List<string>(Current.EquippedBySlot.Keys);
            foreach (var slot in slots)
            {
                if (Current.EquippedBySlot[slot] == itemId)
                {
                    Current.EquippedBySlot.Remove(slot);
                    unequipped = true;
                }
            }

            if (!removed && !unequipped) return ServiceResult.Ok(); // no change, no event

            var save = await SaveAsync();
            if (save.Success)
            {
                if (removed) InventoryChanged?.Invoke();
                if (unequipped) EquipmentChanged?.Invoke();
            }
            return save;
        }

        public async Task<ServiceResult> AddCoinsAsync(int amount)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (amount < 0) return ServiceResult.Fail("amount must be non-negative.");
            if (amount == 0) return ServiceResult.Ok(); // no change, no event

            Current.Coins += amount;

            var save = await SaveAsync();
            if (save.Success) CoinsChanged?.Invoke();
            return save;
        }

        public async Task<ServiceResult> TrySpendCoinsAsync(int amount)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (amount < 0) return ServiceResult.Fail("amount must be non-negative.");
            if (Current.Coins < amount) return ServiceResult.Fail("Insufficient coins.");
            if (amount == 0) return ServiceResult.Ok(); // no change, no event

            Current.Coins -= amount;

            var save = await SaveAsync();
            if (save.Success) CoinsChanged?.Invoke();
            return save;
        }

        public async Task<ServiceResult> EquipAsync(string slot, string itemId)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (string.IsNullOrEmpty(slot)) return ServiceResult.Fail("slot is required.");
            if (string.IsNullOrEmpty(itemId)) return ServiceResult.Fail("itemId is required.");
            if (!Current.Owns(itemId)) return ServiceResult.Fail("Cannot equip an item the player does not own.");

            if (Current.EquippedBySlot.TryGetValue(slot, out string equipped) && equipped == itemId)
                return ServiceResult.Ok();

            Current.EquippedBySlot[slot] = itemId;
            var save = await SaveAsync();
            if (save.Success) EquipmentChanged?.Invoke();
            return save;
        }
    }
}
