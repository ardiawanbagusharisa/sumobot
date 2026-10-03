using System.Linq;
using SumoServices;
using UnityEngine;

// Owns the Market screen's chat / inventory panel toggles only. The item-detail panel is no
// longer managed here — ItemDetailController is its sole owner (visibility + content + buy).
// Kept in the scene because ToggleChat / ToggleInventory are bound to live scene onClicks.
public class MarketManager : MonoBehaviour
{
    public GameObject PanelChat;
    public GameObject PanelInventory;
    public GameObject buttonUnfoldChat;
    public GameObject buttonUnfoldInventory;

    private MarketChatController chatController;

    private void Awake()
    {
        EnsureChatController();
    }

	public void ToggleChat() {
        SFXManager.Instance.Play2D("ui_accept");
		bool isActive = PanelChat.activeSelf;
        PanelChat.SetActive(!isActive);
        buttonUnfoldChat.SetActive(isActive);
		if (!isActive)
			EnsureChatController()?.RefreshNow();
	}

    public static void OpenChatFor(CatalogItem item)
    {
        MarketManager manager = Resources.FindObjectsOfTypeAll<MarketManager>()
            .FirstOrDefault(candidate => candidate.gameObject.scene.IsValid() && candidate.gameObject.activeInHierarchy);
        if (manager == null || manager.PanelChat == null)
        {
            Logger.Warning("[MarketChat] No active Market chat panel was found.");
            return;
        }

        manager.PanelChat.SetActive(true);
        if (manager.buttonUnfoldChat != null)
            manager.buttonUnfoldChat.SetActive(false);

        string creator = string.IsNullOrWhiteSpace(item?.Creator) ? "creator" : item.Creator;
        string itemName = string.IsNullOrWhiteSpace(item?.DisplayName) ? "this item" : item.DisplayName;
        manager.EnsureChatController()?.SetDraft($"@{creator} About {itemName}: ");
    }

    private MarketChatController EnsureChatController()
    {
        if (PanelChat == null)
            return null;
        if (chatController == null)
            chatController = PanelChat.GetComponent<MarketChatController>() ??
                PanelChat.AddComponent<MarketChatController>();
        chatController.Initialize();
        return chatController;
    }

    public void ToggleInventory() {
        SFXManager.Instance.Play2D("ui_accept");
		bool isActive = PanelInventory.activeSelf;
		PanelInventory.SetActive(!isActive);
		buttonUnfoldInventory.SetActive(isActive);
	}
}
