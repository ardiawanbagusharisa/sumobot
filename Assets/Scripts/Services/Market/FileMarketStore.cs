using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine;

namespace SumoServices
{
    /// <summary>
    /// Default IMarketStore: one shared JSON file under Application.persistentDataPath/Market.
    /// Holds the file I/O that used to live inline in LocalMarketService, so runtime behaviour
    /// is unchanged. Exceptions propagate to the caller (LocalMarketService), which wraps them
    /// into a failing ServiceResult exactly as before.
    /// </summary>
    public class FileMarketStore : IMarketStore
    {
        private static string Dir => Path.Combine(Application.persistentDataPath, "Market");
        private static string ListingsPath => Path.Combine(Dir, "listings.json");

        public List<MarketListing> Load()
        {
            if (!File.Exists(ListingsPath)) return null;
            string json = File.ReadAllText(ListingsPath);
            return JsonConvert.DeserializeObject<List<MarketListing>>(json) ?? new List<MarketListing>();
        }

        public void Save(IReadOnlyList<MarketListing> listings)
        {
            Directory.CreateDirectory(Dir);
            File.WriteAllText(ListingsPath, JsonConvert.SerializeObject(listings, Formatting.Indented));
        }
    }
}
