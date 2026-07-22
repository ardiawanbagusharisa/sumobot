using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Owns the Store screen's Shop / Community tab split. Both tabs render into the SAME
// scroll view (Shop and Community share one ItemCellView-populated content list) rather
// than each owning a separate cloned panel — MarketListController (Shop) and
// CommunityListController (Community) both target that shared content, and this
// controller decides which one is allowed to populate it at a time. Attached alongside
// MarketManager (which owns the separate PanelChat toggle — different concern, no conflict).
//
// Toggle wiring is done in code (not Inspector OnValueChanged) so the bool signature
// cannot be silently broken by a Void-mode persistent call in the scene.
public class StoreTabController : MonoBehaviour
{
    [SerializeField] private MarketListController shop;
    [SerializeField] private CommunityListController community;
    [SerializeField] private Toggle shopToggle;
    [SerializeField] private Toggle communityToggle;

    [SerializeField] private TMP_Text subtitle;
    [SerializeField] private string shopTitle = "SHOP";
    [SerializeField] private string communityTitle = "COMMUNITY";

    void OnEnable()
    {
        if (shopToggle != null)
            shopToggle.onValueChanged.AddListener(OnShopToggled);
        if (communityToggle != null)
            communityToggle.onValueChanged.AddListener(OnCommunityToggled);
    }

    void OnDisable()
    {
        if (shopToggle != null)
            shopToggle.onValueChanged.RemoveListener(OnShopToggled);
        if (communityToggle != null)
            communityToggle.onValueChanged.RemoveListener(OnCommunityToggled);
    }

    void Start()
    {
        // Force Shop as the default tab. Setting isOn fires the listener when it changes;
        // if already on, call through directly so the list still populates.
        if (shopToggle != null && !shopToggle.isOn)
            shopToggle.isOn = true;
        else
            ShowShop();
    }

    void OnShopToggled(bool isOn)
    {
        if (isOn) ShowShop();
    }

    void OnCommunityToggled(bool isOn)
    {
        if (isOn) ShowCommunity();
    }

    void ShowShop()
    {
        SFXManager.Instance?.Play2D("ui_accept");
        SetSubtitle(shopTitle);
        community.SetActiveTab(false);
        shop.Populate();
    }

    void ShowCommunity()
    {
        SFXManager.Instance?.Play2D("ui_accept");
        SetSubtitle(communityTitle);
        community.SetActiveTab(true); // populates internally
    }

    void SetSubtitle(string title)
    {
        if (subtitle != null)
            subtitle.text = title;
    }
}
