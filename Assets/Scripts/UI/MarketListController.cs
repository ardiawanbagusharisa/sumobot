using System.Linq;
using SumoServices;
using UnityEngine;

// Populates the Store's shared item list from GameServices.Catalog, filtered to Official
// items only — Player-authored items populate the same list instead via
// CommunityListController when the Community tab is active (see StoreTabController, which
// owns switching between the two and avoids both populating at once). Attach alongside
// StoreTabController; assign Item Cell Prefab in the Inspector (Assets/Prefabs/ItemCell.prefab).
public class MarketListController : MonoBehaviour
{
    [SerializeField] private ItemCellView itemCellPrefab;
    [SerializeField] private ItemDetailController itemDetail;
    [SerializeField] private RectTransform content;

    /// <summary>Rebuilds the shared content list from the Official catalog. Called by StoreTabController.ShowShop.</summary>
    public void Populate()
    {
        for (int i = content.childCount - 1; i >= 0; i--)
            Destroy(content.GetChild(i).gameObject);

        foreach (CatalogItem item in GameServices.Catalog.AllItems.Where(i => i.Source != "Player"))
        {
            ItemCellView cell = Instantiate(itemCellPrefab, content);
            cell.Bind(item);
            if (itemDetail != null)
                cell.Clicked += itemDetail.Show;
        }
    }
}
