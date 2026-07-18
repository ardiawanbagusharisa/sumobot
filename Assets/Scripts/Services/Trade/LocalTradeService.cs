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
        // TODO(economy): sell refund ratio is a product decision. Full refund (100%) for
        // now; make this a per-item or global setting once the PM defines resale value.
        private const float SellRefundRatio = 1.0f;

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

        public async Task<ServiceResult> SellAsync(string itemId)
        {
            if (playerData.Current == null) return ServiceResult.Fail("No player loaded.");

            var item = catalog.GetById(itemId);
            if (item == null) return ServiceResult.Fail($"Unknown item '{itemId}'.");
            if (!playerData.Current.Owns(itemId)) return ServiceResult.Fail("Cannot sell an item the player does not own.");

            var revoke = await playerData.RevokeItemAsync(itemId);
            if (!revoke.Success) return revoke;

            int refund = (int)(item.Price * SellRefundRatio);
            return await playerData.AddCoinsAsync(refund);
        }
    }
}
