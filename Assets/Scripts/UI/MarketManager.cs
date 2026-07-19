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

	public void ToggleChat() {
        SFXManager.Instance.Play2D("ui_accept");
		bool isActive = PanelChat.activeSelf;
        PanelChat.SetActive(!isActive);
        buttonUnfoldChat.SetActive(isActive);
	}

    public void ToggleInventory() {
        SFXManager.Instance.Play2D("ui_accept");
		bool isActive = PanelInventory.activeSelf;
		PanelInventory.SetActive(!isActive);
		buttonUnfoldInventory.SetActive(isActive);
	}
}
