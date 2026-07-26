using System.Collections.Generic;

namespace SumoBot.Graph.Authoring
{
    /// <summary>
    /// Persistence seam for editable bot-graph drafts (E3.1) — the data layer behind the My
    /// Creations list (E3.4). Mirrors <c>IPlayerCatalogStore</c> / <c>IMarketStore</c>: file I/O is
    /// split out so the editor and list logic can be unit-tested against an in-memory store, and a
    /// networked store can replace <see cref="FileDraftStore"/> later behind this interface.
    ///
    /// Drafts are the author's own local work, keyed by a stable <see cref="GraphDraft.Id"/>
    /// (independent of display name). They are NOT catalog items — publishing (E5) is what turns a
    /// draft into an owned, sellable item.
    /// </summary>
    public interface IDraftStore
    {
        /// <summary>Every stored draft. Empty when nothing has been saved yet (first run).</summary>
        IReadOnlyList<GraphDraft> LoadAll();

        /// <summary>The draft with this Id, or null if there is none.</summary>
        GraphDraft Load(string id);

        /// <summary>Persist a draft, replacing any existing one with the same Id (upsert).</summary>
        void Save(GraphDraft draft);

        /// <summary>Remove the draft with this Id; a no-op if it does not exist.</summary>
        void Delete(string id);
    }
}
