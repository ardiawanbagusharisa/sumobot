using SumoServices;
using UnityEngine;

// Populates the Market's Scroll View / Content at runtime from GameServices.Catalog,
// replacing the Item (1)..Item (25) placeholder rows that used to be baked into
// MainMenu.unity. Attach to MenuMarket; assign Item Cell Prefab in the Inspector
// (Assets/Prefabs/ItemCell.prefab).
public class MarketListController : MonoBehaviour
{
    [SerializeField] private ItemCellView itemCellPrefab;
    [SerializeField] private ItemDetailController itemDetail;
    [SerializeField] private RectTransform content;

    void OnEnable()
    {
        Populate();
    }

    private void Populate()
    {
        for (int i = content.childCount - 1; i >= 0; i--)
            Destroy(content.GetChild(i).gameObject);

        foreach (CatalogItem item in GameServices.Catalog.AllItems)
        {
            ItemCellView cell = Instantiate(itemCellPrefab, content);
            cell.Bind(item);
            if (itemDetail != null)
                cell.Clicked += itemDetail.Show;
        }
    }
}
