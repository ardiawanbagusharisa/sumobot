using SumoServices;
using UnityEngine;

// Populates the Inventory panel's Scroll View / Content with the items the player owns,
// resolving PlayerData.OwnedItemIds against the Catalog. It rebuilds itself whenever the
// owned set changes (IPlayerDataService.InventoryChanged) — so an item bought in the
// Market appears here without the buy code knowing this panel exists. Mirrors
// MarketListController; reuses the same ItemCell prefab.
public class InventoryController : MonoBehaviour
{
    [SerializeField] private ItemCellView itemCellPrefab;
    [SerializeField] private RectTransform content;
    [SerializeField] private ItemDetailController itemDetail; // click an owned item -> shared detail panel

    private IPlayerDataService subscribed;

    void OnEnable()
    {
        subscribed = GameServices.PlayerData;
        if (subscribed != null)
            subscribed.InventoryChanged += Rebuild;
        Rebuild();
    }

    void OnDisable()
    {
        if (subscribed != null)
        {
            subscribed.InventoryChanged -= Rebuild;
            subscribed = null;
        }
    }

    private void Rebuild()
    {
        if (content == null) return;

        for (int i = content.childCount - 1; i >= 0; i--)
            Destroy(content.GetChild(i).gameObject);

        var owned = GameServices.PlayerData?.Current?.OwnedItemIds;
        if (owned == null) return;

        foreach (string id in owned)
        {
            CatalogItem item = GameServices.Catalog?.GetById(id);
            if (item == null) continue; // owned id with no catalog definition — skip defensively

            ItemCellView cell = Instantiate(itemCellPrefab, content);
            cell.Bind(item);
            if (itemDetail != null)
                cell.Clicked += itemDetail.Show;
        }
    }
}
