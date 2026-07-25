using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace SumoServices
{
    /// <summary>
    /// Default IPlayerCatalogStore: one shared JSON file under
    /// Application.persistentDataPath/PlayerCatalog, kept separate from the read-only bundled
    /// Resources/Catalog/items.json (see decision-7).
    ///
    /// Write serializes by runtime type (Newtonsoft's default), so a SkinItem's PartSprite /
    /// BotScriptItem's WinRate and the "Type" discriminator are all emitted — NOT via
    /// CatalogItemConverter, whose WriteJson would recurse. Read uses CatalogItemConverter so
    /// the "Type" field picks the concrete subtype back rather than the base CatalogItem.
    ///
    /// A missing file is a clean first run (null). A corrupt/unreadable file degrades to null
    /// and logs, so the catalog can still load the official items rather than crashing startup.
    /// </summary>
    public class FilePlayerCatalogStore : IPlayerCatalogStore
    {
        private static string Dir => Path.Combine(Application.persistentDataPath, "PlayerCatalog");
        private static string ItemsPath => Path.Combine(Dir, "items.json");

        public List<CatalogItem> Load()
        {
            if (!File.Exists(ItemsPath)) return null;
            try
            {
                string json = File.ReadAllText(ItemsPath);
                return JsonConvert.DeserializeObject<List<CatalogItem>>(json, new CatalogItemConverter())
                       ?? new List<CatalogItem>();
            }
            catch (System.Exception e)
            {
                Logger.Error($"[PlayerCatalog] Failed to read {ItemsPath}, ignoring player items: {e.Message}");
                return null;
            }
        }

        public void Save(IReadOnlyList<CatalogItem> items)
        {
            Directory.CreateDirectory(Dir);
            // No converter on write: Newtonsoft serializes each element by its runtime type,
            // which emits subtype fields + the "Type" discriminator. Passing the converter here
            // would invoke its recursive WriteJson.
            File.WriteAllText(ItemsPath, JsonConvert.SerializeObject(items, Formatting.Indented));
        }
    }
}
