using System.Collections.Generic;
using System.Threading.Tasks;
using Newtonsoft.Json;
using UnityEngine;

namespace SumoServices
{
    /// <summary>
    /// Local catalog implementation. Loads item definitions from a JSON TextAsset at
    /// Resources/Catalog/items.json if present; otherwise falls back to a small built-in
    /// set derived from the existing costume parts (Wheel/Eye/Accessory). Designers can
    /// grow the catalog by editing the JSON without any code change.
    /// </summary>
    public class LocalCatalogService : ICatalogService
    {
        private const string ResourcePath = "Catalog/items";

        private List<CatalogItem> items = new();
        private Dictionary<string, CatalogItem> byId = new();

        public IReadOnlyList<CatalogItem> AllItems => items;

        public Task<ServiceResult> LoadAsync()
        {
            items = LoadFromResources() ?? BuildDefaultCatalog();

            byId = new Dictionary<string, CatalogItem>();
            foreach (var item in items)
            {
                if (string.IsNullOrEmpty(item.Id) || byId.ContainsKey(item.Id))
                {
                    Logger.Warning($"[Catalog] Skipping item with missing/duplicate id: '{item.Id}'");
                    continue;
                }
                byId[item.Id] = item;
            }

            Logger.Info($"[Catalog] Loaded {byId.Count} items.");
            return Task.FromResult(ServiceResult.Ok());
        }

        public CatalogItem GetById(string itemId)
        {
            if (string.IsNullOrEmpty(itemId)) return null;
            return byId.TryGetValue(itemId, out var item) ? item : null;
        }

        private List<CatalogItem> LoadFromResources()
        {
            var textAsset = Resources.Load<TextAsset>(ResourcePath);
            if (textAsset == null) return null;

            try
            {
                return JsonConvert.DeserializeObject<List<CatalogItem>>(textAsset.text);
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
