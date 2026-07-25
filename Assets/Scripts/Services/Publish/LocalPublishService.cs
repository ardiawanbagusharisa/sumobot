using System;
using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Local implementation of IPublishService. Orchestration only: composes the catalog
    /// write seam (IPlayerCatalogWriter) and IPlayerDataService, exactly the two dependencies
    /// publish needs, mirroring how LocalMarketService composes catalog + player data.
    /// </summary>
    public class LocalPublishService : IPublishService
    {
        // Namespace prefix guarantees a player id can never collide with a Resources/items.json
        // id (see decision-7); it also makes player-authored ids self-identifying in saves/logs.
        private const string IdPrefix = "player_";

        private readonly IPlayerCatalogWriter catalog;
        private readonly IPlayerDataService playerData;

        public LocalPublishService(IPlayerCatalogWriter catalog, IPlayerDataService playerData)
        {
            this.catalog = catalog;
            this.playerData = playerData;
        }

        public async Task<ServiceResult<string>> PublishAsync(CatalogItem draft)
        {
            if (playerData.Current == null) return ServiceResult<string>.Fail("No player loaded.");
            if (draft == null) return ServiceResult<string>.Fail("draft is required.");

            // Bot-script gate: buyers execute authored code, so listing untrusted scripts is
            // gated on the execution sandbox (A4 / TASK-11 AC#3). Skins have no code and pass.
            if (draft is BotScriptItem)
                return ServiceResult<string>.Fail("Bot script publishing is pending the script sandbox.");

            string id = IdPrefix + Guid.NewGuid().ToString("N");
            draft.Id = id;
            draft.Source = "Player";
            draft.Author = playerData.Current.PlayerId;
            draft.IsDefault = false; // a published creation is never a free default part
            // Stamp the discriminator so CatalogItemConverter round-trips the concrete subtype
            // (Skin's PartSprite, BotScript's WinRate) through the store instead of dropping to
            // the base CatalogItem on read.
            draft.Type = draft switch
            {
                SkinItem _ => "Skin",
                BotScriptItem _ => "BotScript",
                _ => draft.Type
            };

            var add = catalog.Add(draft);
            if (!add.Success) return ServiceResult<string>.Fail(add.Error);

            var grant = await playerData.GrantItemAsync(id);
            if (!grant.Success)
            {
                // Two-phase op (persist item, then grant ownership). On grant failure roll the
                // catalog entry back so we don't leave an item the author doesn't own. Same MVP
                // stance as LocalMarketService.BuyAsync: a rare failure of the compensation
                // itself is logged, not transactional (see decision-7).
                catalog.Remove(id);
                return ServiceResult<string>.Fail(grant.Error);
            }

            return ServiceResult<string>.Ok(id);
        }
    }
}
