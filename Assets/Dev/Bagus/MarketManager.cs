using UnityEngine;

public class MarketManager : MonoBehaviour
{
    public GameObject PanelItemDetail;
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

	public void ShowItemDetail() {
		SFXManager.Instance.Play2D("ui_accept");
		PanelItemDetail.SetActive(true);
    }
	public void HideItemDetail() {
		SFXManager.Instance.Play2D("ui_accept");
		PanelItemDetail.SetActive(false);
	}
}
