namespace SumoServices
{
    /// <summary>
    /// Runtime write side of the catalog for player-published items. Deliberately kept
    /// separate from the read-only <see cref="ICatalogService"/> so publishing never adds
    /// write methods to the read API (see decision-7). LocalCatalogService implements both:
    /// the read interface for browse/lookup, this one for the publish flow.
    ///
    /// Add both indexes the item in the in-memory read model (so it is immediately resolvable
    /// via ICatalogService.GetById) and persists it via the IPlayerCatalogStore; Remove is the
    /// rollback path used when a publish's later step (granting ownership) fails.
    /// </summary>
    public interface IPlayerCatalogWriter
    {
        /// <summary>
        /// Register a player-authored item: index it and persist it. Fails if the item is
        /// null, has no id, or the id collides with an existing catalog item.
        /// </summary>
        ServiceResult Add(CatalogItem item);

        /// <summary>Remove a previously added player-authored item (rollback). No-op if unknown.</summary>
        void Remove(string itemId);
    }
}
