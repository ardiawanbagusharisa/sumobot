using System.Collections.Generic;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

namespace SumoServices
{
    /// <summary>
    /// Local catalog implementation. Loads official item definitions from a JSON TextAsset at
    /// Resources/Catalog/items.json if present; otherwise falls back to a small built-in
    /// set derived from the existing costume parts (Wheel/Eye/Accessory). Designers can
    /// grow the official catalog by editing the JSON without any code change.
    ///
    /// Player-authored (Source=Player) items are NOT in Resources — they live in a writable
    /// <see cref="IPlayerCatalogStore"/> and are merged into the same read model at LoadAsync
    /// (see decision-7). The read surface (ICatalogService) stays read-only; the publish flow
    /// writes through <see cref="IPlayerCatalogWriter"/>, which this class also implements.
    /// </summary>
    public class LocalCatalogService : ICatalogService, IPlayerCatalogWriter
    {
        private const string ResourcePath = "Catalog/items";

        private readonly IPlayerCatalogStore playerStore;

        private List<CatalogItem> items = new();
        private Dictionary<string, CatalogItem> byId = new();

        // The player-authored subset, tracked separately so Add/Remove can persist exactly the
        // player items (never the official Resources rows) back through the store.
        private List<CatalogItem> playerItems = new();

        public IReadOnlyList<CatalogItem> AllItems => items;

        // playerStore defaults to the file-backed implementation so runtime wiring in
        // GameServices (new LocalCatalogService()) needs no store argument; tests inject an
        // in-memory store to exercise the merge/publish path without touching disk.
        public LocalCatalogService(IPlayerCatalogStore playerStore = null)
        {
            this.playerStore = playerStore ?? new FilePlayerCatalogStore();
        }

        public Task<ServiceResult> LoadAsync()
        {
            items = LoadFromResources() ?? BuildDefaultCatalog();

            byId = new Dictionary<string, CatalogItem>();
            foreach (var item in items)
                Index(item);

            // Merge player-authored items on top of the official set. A player id can never
            // collide with a Resources id (they are minted "player_<guid>", see decision-7),
            // so Index's dup-id guard is only a defensive backstop here.
            playerItems = playerStore.Load() ?? new List<CatalogItem>();
            foreach (var item in playerItems)
            {
                if (Index(item)) items.Add(item);
            }

            Logger.Info($"[Catalog] Loaded {byId.Count} items ({playerItems.Count} player-authored).");
            return Task.FromResult(ServiceResult.Ok());
        }

        public CatalogItem GetById(string itemId)
        {
            if (string.IsNullOrEmpty(itemId)) return null;
            return byId.TryGetValue(itemId, out var item) ? item : null;
        }

        public ServiceResult Add(CatalogItem item)
        {
            if (item == null) return ServiceResult.Fail("item is required.");
            if (string.IsNullOrEmpty(item.Id)) return ServiceResult.Fail("item.Id is required.");
            if (byId.ContainsKey(item.Id)) return ServiceResult.Fail($"An item with id '{item.Id}' already exists.");

            Index(item);
            items.Add(item);
            playerItems.Add(item);
            playerStore.Save(playerItems);
            return ServiceResult.Ok();
        }

        public void Remove(string itemId)
        {
            if (string.IsNullOrEmpty(itemId) || !byId.ContainsKey(itemId)) return;

            byId.Remove(itemId);
            items.RemoveAll(i => i.Id == itemId);
            playerItems.RemoveAll(i => i.Id == itemId);
            playerStore.Save(playerItems);
        }

        // Adds the item to the id index; returns false (and warns) on a missing/duplicate id so
        // callers can skip adding it to the items list too.
        private bool Index(CatalogItem item)
        {
            if (string.IsNullOrEmpty(item.Id) || byId.ContainsKey(item.Id))
            {
                Logger.Warning($"[Catalog] Skipping item with missing/duplicate id: '{item.Id}'");
                return false;
            }
            byId[item.Id] = item;
            return true;
        }

        private List<CatalogItem> LoadFromResources()
        {
            var textAsset = Resources.Load<TextAsset>(ResourcePath);
            if (textAsset == null) return null;

            try
            {
                return JsonConvert.DeserializeObject<List<CatalogItem>>(textAsset.text, new CatalogItemConverter());
            }
            catch (System.Exception e)
            {
                Logger.Error($"[Catalog] Failed to parse {ResourcePath}.json, using defaults: {e.Message}");
                return null;
            }
        }

        // Mirrors the parts the game already ships (see PlayerProfile.PrepareParts).
        private List<CatalogItem> BuildDefaultCatalog()
        {
            // Only the parts the game already ships. No invented "premium" items or
            // prices — extend this (or Resources/Catalog/items.json) once real art and
            // an acquisition model exist.
            return new List<CatalogItem>
            {
                new() { Id = "wheel_1",     DisplayName = "Standard Wheel", Slot = "Wheel",     IsDefault = true, IconResourcePath = "Sprites/Character/Wheel_1" },
                new() { Id = "eye_1",       DisplayName = "Standard Eye",   Slot = "Eye",       IsDefault = true, IconResourcePath = "Sprites/Character/Eye_1" },
                new() { Id = "accessory_1", DisplayName = "No Accessory",   Slot = "Accessory", IsDefault = true, IconResourcePath = "Sprites/Character/Accessory_1" },
            };
        }
    }
}
