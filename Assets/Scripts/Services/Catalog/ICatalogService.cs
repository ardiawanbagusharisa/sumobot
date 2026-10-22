using System.Collections.Generic;
using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Read-only source of item definitions for the Market and Inventory. The catalog
    /// itself is not player-specific — purchases and ownership live in PlayerData. A
    /// later UGS Economy implementation can replace the local one behind this interface.
    /// </summary>
    public interface ICatalogService
    {
        /// <summary>Load the catalog into memory. Call once at startup.</summary>
        Task<ServiceResult> LoadAsync();

        /// <summary>All items (Market listing).</summary>
        IReadOnlyList<CatalogItem> AllItems { get; }

        /// <summary>Look up a single item by id, or null if unknown.</summary>
        CatalogItem GetById(string itemId);
    }
}
