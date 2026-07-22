using SumoServices;
using UnityEngine;

// Populates the Store's shared item list from GameServices.Market.ActiveListings —
// player-to-player listings — when the Community tab is active. Shares the same
// ItemCellView-populated content RectTransform as MarketListController (the Shop tab);
// StoreTabController owns which one is currently allowed to (re)populate it, so only the
// active tab reacts to its data source changing. Reuses the same ItemCell prefab, bound
// with the listing's price rather than the item's Price.
public class CommunityListController : MonoBehaviour
{
    [SerializeField] private ItemCellView itemCellPrefab;
    [SerializeField] private ItemDetailController itemDetail;
    [SerializeField] private RectTransform content;

    private IMarketService subscribed;
    private bool isActiveTab;

    void OnEnable()
    {
        subscribed = GameServices.Market;
        if (subscribed != null)
            subscribed.ListingsChanged += OnListingsChanged;
    }

    void OnDisable()
    {
        if (subscribed != null)
        {
            subscribed.ListingsChanged -= OnListingsChanged;
            subscribed = null;
        }
        isActiveTab = false;
    }

    /// <summary>Called by StoreTabController when the Community tab becomes the active/inactive one.</summary>
    public void SetActiveTab(bool active)
    {
        isActiveTab = active;
        if (isActiveTab) Populate();
    }

    private void OnListingsChanged()
    {
        if (isActiveTab) Populate();
    }

    private void Populate()
    {
        if (content == null) return;

        for (int i = content.childCount - 1; i >= 0; i--)
            Destroy(content.GetChild(i).gameObject);

        var market = GameServices.Market;
        if (market == null) return;

        foreach (MarketListing listing in market.ActiveListings)
        {
            CatalogItem item = GameServices.Catalog?.GetById(listing.ItemId);
            if (item == null) continue; // listing with no resolvable catalog item — skip defensively

            ItemCellView cell = Instantiate(itemCellPrefab, content);
            cell.Bind(item, listing);
            if (itemDetail != null)
                cell.ClickedWithListing += itemDetail.Show;
        }
    }
}
