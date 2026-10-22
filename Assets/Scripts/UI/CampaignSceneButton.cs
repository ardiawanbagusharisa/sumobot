using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

/// <summary>Attach directly to a MenuCampaign ButtonCampaign object.</summary>
[RequireComponent(typeof(Button))]
public class CampaignSceneButton : MonoBehaviour
{
    [SerializeField] private string sceneName;

    private void Awake()
    {
        Button button = GetComponent<Button>();
        button.onClick = new Button.ButtonClickedEvent();
        button.onClick.AddListener(() => SceneManager.LoadScene(sceneName));
    }
}
