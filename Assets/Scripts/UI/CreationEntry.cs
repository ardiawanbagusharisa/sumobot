using SumoServices;
using SumoBot.Graph.Authoring;
using UnityEngine;

// One row in the My Creations list (decision-10). A creation is either an editable draft
// (GraphDraft, from the draft store) or a published bot (a Player-authored BotScriptItem from
// the catalog). Those are different types in different assemblies, so this wrapper gives the
// cell and detail views a single shape to bind against and a Kind to branch on (Draft shows
// Edit/Publish/Delete; Published is immutable and shows List/Duplicate).
public class CreationEntry
{
    public enum Kind { Draft, Published }

    public Kind Type { get; private set; }
    public GraphDraft Draft { get; private set; }        // set when Type == Draft
    public CatalogItem Published { get; private set; }   // set when Type == Published

    public static CreationEntry ForDraft(GraphDraft draft)
        => new() { Type = Kind.Draft, Draft = draft };

    public static CreationEntry ForPublished(CatalogItem item)
        => new() { Type = Kind.Published, Published = item };

    /// <summary>Stable id — draft id or catalog item id.</summary>
    public string Id => Type == Kind.Draft ? Draft?.Id : Published?.Id;

    public string DisplayName => Type == Kind.Draft
        ? (string.IsNullOrEmpty(Draft?.Name) ? "Untitled bot" : Draft.Name)
        : Published?.DisplayName;

    /// <summary>Optional long text (published bots only; drafts carry just a name).</summary>
    public string Description => Type == Kind.Published ? Published?.Description : null;

    /// <summary>Resources icon path (published only); null for a draft, which keeps the cell's default icon.</summary>
    public string IconResourcePath => Type == Kind.Published ? Published?.IconResourcePath : null;

    /// <summary>Tint parsed from the published item's IconColor; white when absent.</summary>
    public bool TryGetIconColor(out Color color)
    {
        string hex = Type == Kind.Published ? Published?.IconColor : null;
        if (!string.IsNullOrEmpty(hex) && ColorUtility.TryParseHtmlString(hex, out color))
            return true;
        color = Color.white;
        return false;
    }
}
