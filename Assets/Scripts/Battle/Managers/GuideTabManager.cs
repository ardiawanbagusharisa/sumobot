using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class GuideTabManager : MonoBehaviour
{
    [Header("Tab Buttons")]
    public Button gameplayTab;
    public Button rulesTab;
    public Button controlsTab;

    [Header("Content Panel")]
    public GameObject guidePanel;      
    public TMP_Text titleText;         
    public TMP_Text contentText;       

    [Header("Scroll View")]
    public ScrollRect scrollRect;  // Drag ScrollRect-mu ke sini

    [Header("Tab Highlight Colors")]
    public Color activeTabColor = new Color(1f, 0.89f, 0.62f);   
    public Color inactiveTabColor = new Color(0.88f, 0.88f, 0.88f);

    [TextArea(2,6)] public string gameplayContent;
    [TextArea(2,6)] public string rulesContent;
    [TextArea(2,6)] public string controlsContent;

    void Start()
    {
        gameplayTab.onClick.AddListener(() => ShowTab("Gameplay"));
        rulesTab.onClick.AddListener(() => ShowTab("Rules"));
        controlsTab.onClick.AddListener(() => ShowTab("Controls"));

        if (guidePanel != null) guidePanel.SetActive(false);
    }

    public void ShowPanel()
    {
        if (guidePanel != null)
        {
            guidePanel.SetActive(true);
            ShowTab("Gameplay");
        }
    }

    public void HidePanel()
    {
        if (guidePanel != null)
            guidePanel.SetActive(false);
    }

    public void ShowTab(string tab)
    {
        switch (tab)
        {
            case "Gameplay":
                titleText.text = "Gameplay";
                contentText.text = gameplayContent;
                SetTabHighlight(gameplayTab, true);
                SetTabHighlight(rulesTab, false);
                SetTabHighlight(controlsTab, false);
                break;
            case "Rules":
                titleText.text = "Rules";
                contentText.text = rulesContent;
                SetTabHighlight(gameplayTab, false);
                SetTabHighlight(rulesTab, true);
                SetTabHighlight(controlsTab, false);
                break;
            case "Controls":
                titleText.text = "Controls";
                contentText.text = controlsContent;
                SetTabHighlight(gameplayTab, false);
                SetTabHighlight(rulesTab, false);
                SetTabHighlight(controlsTab, true);
                break;
        }

        // SCROLL TO TOP setiap ganti tab
        if (scrollRect != null)
        {
            // Di-frame berikutnya (biar ContentSizeFitter selesai update size)
            StartCoroutine(ResetScroll());
        }
    }

    private System.Collections.IEnumerator ResetScroll()
    {
        yield return null; // Tunggu 1 frame
        scrollRect.verticalNormalizedPosition = 1f;
    }

    private void SetTabHighlight(Button tab, bool active)
    {
        var colors = tab.colors;
        colors.normalColor = active ? activeTabColor : inactiveTabColor;
        colors.selectedColor = active ? activeTabColor : inactiveTabColor;
        tab.colors = colors;

        Image tabImg = tab.GetComponent<Image>();
        if (tabImg != null)
            tabImg.color = active ? activeTabColor : inactiveTabColor;
    }
}
