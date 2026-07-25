using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Turns a player's creation into an owned, sellable catalog item. This is a deliberately
    /// separate seam from IMarketService (see decision-7): publish makes an item EXIST and
    /// grants the author ownership; listing (IMarketService.ListAsync) offers an owned+authored
    /// item for sale. The "explicit publish" product decision keeps the two lifecycles apart —
    /// a player can publish without ever listing.
    ///
    /// Composes the catalog write seam (IPlayerCatalogWriter) and IPlayerDataService. A later
    /// networked implementation can replace it, mirroring the other Local* services.
    /// </summary>
    public interface IPublishService
    {
        /// <summary>
        /// Publish a draft as a Source=Player item authored by the current player: mints a
        /// globally-unique id, stamps Source/Author, persists it to the player catalog, and
        /// grants the author ownership. Returns the new item id on success.
        ///
        /// Fails if no player is loaded, the draft is null, or (for now) the draft is a bot
        /// script — script publishing is gated until the execution sandbox exists, because a
        /// buyer would run authored code. Skins publish directly.
        /// </summary>
        Task<ServiceResult<string>> PublishAsync(CatalogItem draft);
    }
}
