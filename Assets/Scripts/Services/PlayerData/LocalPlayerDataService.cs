using System;
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
                Current.EquippedBySlot ??= new();

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

            if (!Current.OwnedItemIds.Contains(itemId))
                Current.OwnedItemIds.Add(itemId);

            return await SaveAsync();
        }

        public async Task<ServiceResult> EquipAsync(string slot, string itemId)
        {
            if (Current == null) return ServiceResult.Fail("Nothing loaded.");
            if (string.IsNullOrEmpty(slot)) return ServiceResult.Fail("slot is required.");
            if (!Current.Owns(itemId)) return ServiceResult.Fail("Cannot equip an item the player does not own.");

            Current.EquippedBySlot[slot] = itemId;
            return await SaveAsync();
        }
    }
}
