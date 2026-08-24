using System.Collections.Generic;
using SumoServices;
using SumoBot.Graph.Authoring;
using UnityEngine;
using UnityEngine.UI;

// Populates the Workshop's My Creations grid (decision-10): the player's editable drafts (from
// the draft store) plus their published bots (Player-authored BotScriptItems, Author == self).
// Mirrors MarketListController's rebuild pattern. Clicking a cell opens the shared
// CreationDetailController. Attach to the MenuWorkshop panel so OnEnable refreshes each time the
// pillar is shown.
public class MyCreationsListController : MonoBehaviour
{
    [SerializeField] private CreationCellView cellPrefab;
    [SerializeField] private RectTransform content;        // Scroll View/Viewport/Content (with a GridLayoutGroup)
    [SerializeField] private CreationDetailController detail;
    [SerializeField] private GameObject emptyState;        // shown when the player has no creations yet
    [SerializeField] private Button newBotButton;          // "Create New Bot"

    void Awake()
    {
        if (newBotButton != null) newBotButton.onClick.AddListener(OnNewBot);
        // Refresh the list after a detail action (delete/publish/duplicate) changes the set.
        if (detail != null) detail.Changed += Refresh;
    }

    void OnEnable() => Refresh();

    /// <summary>Clear and rebuild the grid from the draft store + the player's published bots.</summary>
    public void Refresh()
    {
        if (content == null || cellPrefab == null) return;

        for (int i = content.childCount - 1; i >= 0; i--)
            Destroy(content.GetChild(i).gameObject);

        var entries = CollectEntries();

        foreach (var entry in entries)
        {
            CreationCellView cell = Instantiate(cellPrefab, content);
            cell.Bind(entry);
            if (detail != null) cell.Clicked += detail.Show;
        }

        if (emptyState != null) emptyState.SetActive(entries.Count == 0);
    }

    private static List<CreationEntry> CollectEntries()
    {
        var entries = new List<CreationEntry>();

        foreach (GraphDraft draft in DraftStoreProvider.Instance.LoadAll())
            entries.Add(CreationEntry.ForDraft(draft));

        // Published bots authored by the current player (empty until publish, E5, exists).
        string selfId = GameServices.PlayerData?.Current?.PlayerId;
        if (GameServices.Catalog != null && !string.IsNullOrEmpty(selfId))
            foreach (CatalogItem item in GameServices.Catalog.AllItems)
                if (item is BotScriptItem && item.Author == selfId)
                    entries.Add(CreationEntry.ForPublished(item));

        return entries;
    }

    // Open the node editor on a brand-new bot. The draft is persisted only when the author hits
    // Save in the editor (GraphEditorController), so backing out leaves no empty draft behind.
    private void OnNewBot() => WorkshopRouting.OpenEditor();
}
