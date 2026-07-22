using SumoCampaign;
using UnityEngine;
using UnityEngine.UI;

/// <summary>Scene-attached handler for PanelMission > Tab_Controls > Continue.</summary>
[RequireComponent(typeof(Button))]
public class CampaignContinueButton : MonoBehaviour
{
    private void Awake()
    {
        Button button = GetComponent<Button>();
        button.onClick = new Button.ButtonClickedEvent();
        button.onClick.AddListener(() => FindFirstObjectByType<CampaignLevelController>()?.ContinueFromMissionPanel());
    }
}
