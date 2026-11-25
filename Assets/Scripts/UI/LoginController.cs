using System.Threading.Tasks;
using SumoServices;
using SumoMultiplayer;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Drives Unity Authentication sign-in while retaining the authored ready/error UI.
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

    [SerializeField] private GameObject credentialsPanel;
    [SerializeField] private TMP_InputField usernameInput;
    [SerializeField] private TMP_InputField passwordInput;
    [SerializeField] private TMP_Text statusText;
    [SerializeField] private Button signInButton;
    [SerializeField] private Button signUpButton;
    [SerializeField] private TMP_Text signUpLabel;
    [SerializeField] private Button guestButton;
    [SerializeField] private Button upgradeButton;
    private bool linkingGuest;

    private enum State { Credentials, Loading, Ready, Error }

    private bool busy;

    void Awake()
    {
        if (signInButton != null)
            signInButton.onClick.AddListener(() => _ = RunLoginAsync(AccountLoginMode.SignIn));
        if (signUpButton != null)
            signUpButton.onClick.AddListener(() => _ = RunLoginAsync(AccountLoginMode.SignUp));
        if (guestButton != null)
            guestButton.onClick.AddListener(() => _ = RunLoginAsync(AccountLoginMode.Guest));
        if (upgradeButton != null)
            upgradeButton.onClick.AddListener(OnUpgradeGuest);
        if (retryButton != null)
            retryButton.onClick.AddListener(() => SetState(State.Credentials));
        if (continueButton != null)
            continueButton.onClick.AddListener(() => _ = OnContinueAsync());
    }

    private void OnUpgradeGuest()
    {
        linkingGuest = true;
        signInButton.interactable = false;
        signUpLabel.text = "UPGRADE GUEST";
        statusText.text = "Choose a username and password for this guest ID.";
        SetState(State.Credentials);
    }

    void Start()
    {
        if (GameServices.Auth.Current != null)
        {
            ShowReady(GameServices.Auth.Current);
            return;
        }

        SetState(State.Credentials);
        if (OnlineLaunchOptions.HasAutoOnlineFlag())
            _ = RunLoginAsync(AccountLoginMode.Guest);
    }

    private async Task RunLoginAsync(AccountLoginMode mode)
    {
        if (busy) return;
        busy = true;

        SetState(State.Loading);

        ServiceResult<PlayerAccount> result = linkingGuest && mode == AccountLoginMode.SignUp
            ? await GameServices.Auth.LinkUsernamePasswordAsync(usernameInput.text.Trim(), passwordInput.text)
            : await GameManager.Instance.StartSessionAsync(mode, usernameInput.text.Trim(), passwordInput.text);

        if (result.Success)
        {
            passwordInput.text = string.Empty;
            linkingGuest = false;
            signInButton.interactable = true;
            signUpLabel.text = "CREATE ACCOUNT";
            GameManager.Instance.ApplyAccount(result.Value.PlayerId, result.Value.DisplayName);
            ShowReady(result.Value);
        }
        else
        {
            statusText.text = result.Error;
            SetState(State.Credentials);
        }

        busy = false;
    }

    private void ShowReady(PlayerAccount account)
    {
        if (accountInfoText != null)
            accountInfoText.text = $"{(account.IsGuest ? "Guest" : "Account")} ID: {account.PlayerId}";
        if (displayNameInput != null)
            displayNameInput.text = account.DisplayName;
        if (upgradeButton != null)
            upgradeButton.gameObject.SetActive(account.IsGuest);
        SetState(State.Ready);
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
        mainMenu.ShowReturnMenuOrMainMenu();
    }

    private void SetState(State state)
    {
        if (credentialsPanel != null) credentialsPanel.SetActive(state == State.Credentials);
        if (loadingPanel != null) loadingPanel.SetActive(state == State.Loading);
        if (readyPanel != null) readyPanel.SetActive(state == State.Ready);
        if (errorPanel != null) errorPanel.SetActive(state == State.Error);
    }

}
