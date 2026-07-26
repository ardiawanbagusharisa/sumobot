using System;

namespace SumoBot.Graph.Authoring
{
    /// <summary>
    /// One saved, editable bot-graph draft (decision-10). A draft is work-in-progress the author
    /// keeps locally: just a stable <see cref="Id"/> plus the <see cref="BotGraph"/> payload. The
    /// Id is independent of the display name (<see cref="BotGraph.Name"/>) so renaming a draft
    /// never orphans its stored file.
    ///
    /// A draft is deliberately NOT a catalog item: it has no owner, price, or sellable flag.
    /// Publishing (E5) is what freezes a draft into an owned, sellable catalog item; until then it
    /// only ever lives in the draft store as editable JSON.
    /// </summary>
    public class GraphDraft
    {
        public string Id;
        public BotGraph Graph;

        public GraphDraft() { }

        public GraphDraft(string id, BotGraph graph)
        {
            Id = id;
            Graph = graph;
        }

        /// <summary>The draft's display name, taken from the graph (may be null/empty).</summary>
        public string Name => Graph?.Name;

        /// <summary>Wrap a graph as a brand-new draft with a fresh, unique Id.</summary>
        public static GraphDraft NewDraft(BotGraph graph) => new(Guid.NewGuid().ToString("N"), graph);
    }
}
