using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Market transactions: turning coins into owned items. This is the orchestration
    /// layer that ties the read-only catalog (prices) to the player's save (coin balance
    /// + inventory) — it owns no data of its own. The Market is buy-only by product
    /// decision; a later UGS Economy implementation can replace the local one behind this
    /// interface.
    /// </summary>
    public interface ITradeService
    {
        /// <summary>
        /// Buy a catalog item: charge its Price and grant it to the inventory. Fails
        /// (without charging) if the item is unknown, already owned, or unaffordable.
        /// </summary>
        Task<ServiceResult> BuyAsync(string itemId);
    }
}
