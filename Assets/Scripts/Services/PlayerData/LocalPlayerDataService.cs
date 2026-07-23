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

        // The loadout equip/unequip act on. Set to the Left side's loadout on Load; editors
        // repoint it with SelectLoadout (e.g. BotCreator selects the side it opened for).
        private string activeLoadoutId;

        public Loadout ActiveLoadout
        {
            get
            {
                if (Current == null) return null;
                return Current.GetLoadout(activeLoadoutId)
                       ?? (Current.Loadouts != null && Current.Loadouts.Count > 0 ? Current.Loadouts[0] : null);
            }
        }

        public event Action CoinsChanged;
        public event Action InventoryChanged;
        public event Action EquipmentChanged;

        private static string Dir => Path.Combine(Application.persistentDataPath, "PlayerData");

        private static string PathFor(string playerId) => Path.Combine(Dir, $"{playerId}.json");

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

                // Bring pre-loadout saves up to the decision-5 model (idempotent), then edit
                // the Left side's loadout by default.
                Current.MigrateToLoadouts();
                activeLoadoutId = Current.GetLoadoutForSide(PlayerData.SideLeft)?.Id
                                  ?? Current.Loadouts[0].Id;

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
                Directory.CreateDirectory(Dir);
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

            // Ownership is account-wide, so drop the now-unowned item from EVERY loadout that
            // equipped it (not just the active one) — otherwise a loadout could reference an
            // item the account no longer owns.
            bool unequipped = false;
            foreach (var loadout in Current.Loadouts)
            {
                var slots = new List<string>(loadout.EquippedBySlot.Keys);
                foreach (var slot in slots)
                {
                    if (loadout.EquippedBySlot[slot] == itemId)
                    {
                        loadout.EquippedBySlot.Remove(slot);
                        unequipped = true;
                    }
                }
            }

            if (!removed) return ServiceResult.Ok(); // not owned — no change, no event

            var save = await SaveAsync();
            if (save.Success)
            {
                InventoryChanged?.Invoke();
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

        public void SelectLoadout(string loadoutId)
        {
            if (Current == null || string.IsNullOrEmpty(loadoutId)) return;
            if (loadoutId == activeLoadoutId) return;          // already active — no event
            if (Current.GetLoadout(loadoutId) == null) return; // unknown id — ignore

            activeLoadoutId = loadoutId;
            // Switching the edited loadout is a full costume state change from every live view's
            // point of view; reuse EquipmentChanged so preview/tabs/marks rebuild themselves.
            EquipmentChanged?.Invoke();
        }

        public async Task<ServiceResult> EquipAsync(string slot, string itemId)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (string.IsNullOrEmpty(slot)) return ServiceResult.Fail("slot is required.");
            if (!Current.Owns(itemId)) return ServiceResult.Fail("Cannot equip an item the player does not own.");

            var loadout = ActiveLoadout;
            if (loadout == null) return ServiceResult.Fail("No active loadout.");

            if (loadout.EquippedBySlot.TryGetValue(slot, out var existing) && existing == itemId)
                return ServiceResult.Ok(); // already equipped — no change, no event

            loadout.EquippedBySlot[slot] = itemId;
            var save = await SaveAsync();
            if (save.Success) EquipmentChanged?.Invoke();
            return save;
        }

        public async Task<ServiceResult> UnequipAsync(string slot)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (string.IsNullOrEmpty(slot)) return ServiceResult.Fail("slot is required.");

            var loadout = ActiveLoadout;
            if (loadout == null) return ServiceResult.Fail("No active loadout.");

            if (!loadout.EquippedBySlot.Remove(slot))
                return ServiceResult.Ok(); // slot already empty — no change, no event

            var save = await SaveAsync();
            if (save.Success) EquipmentChanged?.Invoke();
            return save;
        }

        public Task<ServiceResult> AddCoinsToPlayerAsync(string playerId, int amount)
        {
            if (string.IsNullOrEmpty(playerId)) return Task.FromResult(ServiceResult.Fail("playerId is required."));
            if (amount < 0) return Task.FromResult(ServiceResult.Fail("amount must be non-negative."));
            if (amount == 0) return Task.FromResult(ServiceResult.Ok()); // no change, no event

            // Crediting the signed-in player themselves stays in memory and fires the
            // normal CoinsChanged event, same as AddCoinsAsync.
            if (Current != null && Current.PlayerId == playerId)
                return AddCoinsAsync(amount);

            try
            {
                string path = PathFor(playerId);
                var target = File.Exists(path)
                    ? JsonConvert.DeserializeObject<PlayerData>(File.ReadAllText(path)) ?? PlayerData.CreateDefault(playerId)
                    : PlayerData.CreateDefault(playerId);
                target.PlayerId = playerId;
                target.Coins += amount;

                Directory.CreateDirectory(Dir);
                File.WriteAllText(path, JsonConvert.SerializeObject(target, Formatting.Indented));
                return Task.FromResult(ServiceResult.Ok());
            }
            catch (Exception e)
            {
                Logger.Error($"[PlayerData] AddCoinsToPlayerAsync({playerId}) failed: {e.Message}");
                return Task.FromResult(ServiceResult.Fail(e.Message));
            }
        }
    }
}
