using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Local implementation of ITradeService. Composes the catalog (for prices) and the
    /// player-data service (for balance + inventory); all persistence happens through the
    /// latter's grant/spend helpers, so this class stays pure orchestration.
    /// </summary>
    public class LocalTradeService : ITradeService
    {
        private readonly ICatalogService catalog;
        private readonly IPlayerDataService playerData;

        public LocalTradeService(ICatalogService catalog, IPlayerDataService playerData)
        {
            this.catalog = catalog;
            this.playerData = playerData;
        }

        public async Task<ServiceResult> BuyAsync(string itemId)
        {
            if (playerData.Current == null) return ServiceResult.Fail("No player loaded.");

            var item = catalog.GetById(itemId);
            if (item == null) return ServiceResult.Fail($"Unknown item '{itemId}'.");
            if (playerData.Current.Owns(itemId)) return ServiceResult.Fail("Item already owned.");

            var spend = await playerData.TrySpendCoinsAsync(item.Price);
            if (!spend.Success) return spend;

            var grant = await playerData.GrantItemAsync(itemId);
            if (!grant.Success)
            {
                // Compensate so a failed grant never silently pockets the player's coins.
                await playerData.AddCoinsAsync(item.Price);
                return grant;
            }

            return ServiceResult.Ok();
        }
    }
}
