using System.Threading.Tasks;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Drives the Login menu state: kicks off GameManager.StartSessionAsync() on scene start.
// Loading -> Ready (shows the signed-in account, lets the player edit their display name,
// waits for Continue) or Loading -> Error (shows the failure with a Retry button).
public class LoginController : MonoBehaviour
{
    [SerializeField] private MainMenuController mainMenu;

    [SerializeField] private GameObject loadingPanel;

    [SerializeField] private GameObject readyPanel;
    [SerializeField] private TextMeshProUGUI accountInfoText;
    [SerializeField] private TMP_InputField displayNameInput;
    [SerializeField] private Button continueButton;

    [SerializeField] private GameObject errorPanel;
    [SerializeField] private TextMeshProUGUI errorText;
    [SerializeField] private Button retryButton;

    private enum State { Loading, Ready, Error }

    private bool busy;

    void Awake()
    {
        if (retryButton != null)
            retryButton.onClick.AddListener(() => _ = RunLoginAsync());
        if (continueButton != null)
            continueButton.onClick.AddListener(() => _ = OnContinueAsync());
    }

    void Start()
    {
        // GameServices is ready before this scene loads.
        // Skip login if a session already exists.
        if (GameServices.Auth.Current != null)
        {
            mainMenu.ShowMainMenu();
            return;
        }

        _ = RunLoginAsync();
    }

    private async Task RunLoginAsync()
    {
        if (busy) return;
        busy = true;

        SetState(State.Loading);

        var result = await GameManager.Instance.StartSessionAsync();

        if (result.Success)
        {
            var account = result.Value;
            GameManager.Instance.ApplyAccount(account.PlayerId, account.DisplayName);
            if (accountInfoText != null)
                accountInfoText.text = $"{account.PlayerId}";
            if (displayNameInput != null)
                displayNameInput.text = account.DisplayName;
            SetState(State.Ready);
        }
        else
        {
            // TODO: replace with a player-facing message once copy is decided —
            // result.Error is a raw backend/technical string, not UI copy.
            if (errorText != null)
                errorText.text = $"Login failed: {result.Error}";
            SetState(State.Error);
        }

        busy = false;
    }

    private async Task OnContinueAsync()
    {
        if (busy) return;
        busy = true;

        var newName = displayNameInput != null ? displayNameInput.text.Trim() : null;
        var current = GameServices.Auth.Current;

        if (!string.IsNullOrEmpty(newName) && current != null && newName != current.DisplayName)
        {
            var rename = await GameServices.Auth.SetDisplayNameAsync(newName);
            if (!rename.Success)
            {
                // TODO: same as above — needs player-facing copy, not the raw error.
                if (errorText != null)
                    errorText.text = $"Could not save name: {rename.Error}";
                SetState(State.Error);
                busy = false;
                return;
            }
            GameManager.Instance.ApplyAccount(current.PlayerId, current.DisplayName);
        }

        busy = false;
        mainMenu.ShowMainMenu();
    }

    private void SetState(State state)
    {
        if (loadingPanel != null) loadingPanel.SetActive(state == State.Loading);
        if (readyPanel != null) readyPanel.SetActive(state == State.Ready);
        if (errorPanel != null) errorPanel.SetActive(state == State.Error);
    }
}
