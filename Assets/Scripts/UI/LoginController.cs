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

    private GameObject credentialsPanel;
    private TMP_InputField usernameInput;
    private TMP_InputField passwordInput;
    private TMP_Text statusText;
    private Button signInButton;
    private TMP_Text signUpLabel;
    private Button upgradeButton;
    private bool linkingGuest;

    private enum State { Credentials, Loading, Ready, Error }

    private bool busy;

    void Awake()
    {
        BuildCredentialsPanel();
        BuildUpgradeButton();
        if (retryButton != null)
            retryButton.onClick.AddListener(() => SetState(State.Credentials));
        if (continueButton != null)
            continueButton.onClick.AddListener(() => _ = OnContinueAsync());
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
        mainMenu.ShowMainMenu();
    }

    private void SetState(State state)
    {
        if (credentialsPanel != null) credentialsPanel.SetActive(state == State.Credentials);
        if (loadingPanel != null) loadingPanel.SetActive(state == State.Loading);
        if (readyPanel != null) readyPanel.SetActive(state == State.Ready);
        if (errorPanel != null) errorPanel.SetActive(state == State.Error);
    }

    private void BuildCredentialsPanel()
    {
        credentialsPanel = new GameObject("CredentialsPanel", typeof(RectTransform), typeof(Image));
        RectTransform backdrop = credentialsPanel.GetComponent<RectTransform>();
        backdrop.SetParent(transform, false);
        backdrop.anchorMin = Vector2.zero;
        backdrop.anchorMax = Vector2.one;
        backdrop.offsetMin = backdrop.offsetMax = Vector2.zero;
        credentialsPanel.GetComponent<Image>().color = new Color(.45f, .55f, .95f);

        GameObject card = new("Account Card", typeof(RectTransform), typeof(Image), typeof(VerticalLayoutGroup));
        RectTransform rect = card.GetComponent<RectTransform>();
        rect.SetParent(backdrop, false);
        rect.anchorMin = rect.anchorMax = new Vector2(.5f, .5f);
        rect.sizeDelta = new Vector2(550f, 520f);
        card.GetComponent<Image>().color = new Color(.19f, .27f, .49f);
        VerticalLayoutGroup layout = card.GetComponent<VerticalLayoutGroup>();
        layout.padding = new RectOffset(40, 40, 20, 20);
        layout.spacing = 7f;
        layout.childControlWidth = true;
        layout.childForceExpandWidth = true;
        layout.childControlHeight = false;
        layout.childForceExpandHeight = false;

        AddText(card.transform, "SUMOBOT ACCOUNT", 30, 52);
        AddText(card.transform, "Sign in, create an account, or play as a guest.", 17, 40);
        usernameInput = AddInput(card.transform, "Username", false);
        passwordInput = AddInput(card.transform, "Password", true);
        signInButton = AddButton(card.transform, "SIGN IN", () => _ = RunLoginAsync(AccountLoginMode.SignIn));
        Button signUp = AddButton(card.transform, "CREATE ACCOUNT", () => _ = RunLoginAsync(AccountLoginMode.SignUp));
        signUpLabel = signUp.GetComponentInChildren<TMP_Text>();
        AddButton(card.transform, "CONTINUE AS GUEST", () => _ = RunLoginAsync(AccountLoginMode.Guest));
        statusText = AddText(card.transform, string.Empty, 17, 58);
        statusText.color = new Color(1f, .82f, .5f);
    }

    private void BuildUpgradeButton()
    {
        if (readyPanel == null) return;
        upgradeButton = AddButton(readyPanel.transform, "LINK GUEST ACCOUNT", () =>
        {
            linkingGuest = true;
            signInButton.interactable = false;
            signUpLabel.text = "UPGRADE GUEST";
            statusText.text = "Choose a username and password for this guest ID.";
            SetState(State.Credentials);
        });
        RectTransform rect = upgradeButton.GetComponent<RectTransform>();
        rect.anchorMin = rect.anchorMax = new Vector2(.5f, 0f);
        rect.pivot = new Vector2(.5f, 0f);
        rect.anchoredPosition = new Vector2(0f, 12f);
        rect.sizeDelta = new Vector2(270f, 48f);
        upgradeButton.gameObject.SetActive(false);
    }

    private TMP_Text AddText(Transform parent, string value, float size, float height)
    {
        GameObject go = new("Text", typeof(RectTransform), typeof(TextMeshProUGUI), typeof(LayoutElement));
        go.transform.SetParent(parent, false);
        go.GetComponent<LayoutElement>().preferredHeight = height;
        TextMeshProUGUI text = go.GetComponent<TextMeshProUGUI>();
        text.font = accountInfoText != null ? accountInfoText.font : TMP_Settings.defaultFontAsset;
        text.text = value;
        text.fontSize = size;
        text.color = Color.white;
        text.alignment = TextAlignmentOptions.Center;
        return text;
    }

    private TMP_InputField AddInput(Transform parent, string hint, bool password)
    {
        GameObject go = new(hint, typeof(RectTransform), typeof(Image), typeof(TMP_InputField), typeof(LayoutElement));
        go.transform.SetParent(parent, false);
        go.GetComponent<Image>().color = Color.white;
        go.GetComponent<LayoutElement>().preferredHeight = 50f;
        RectTransform viewport = new GameObject("Viewport", typeof(RectTransform), typeof(RectMask2D))
            .GetComponent<RectTransform>();
        viewport.SetParent(go.transform, false);
        viewport.anchorMin = Vector2.zero;
        viewport.anchorMax = Vector2.one;
        viewport.offsetMin = new Vector2(14f, 0f);
        viewport.offsetMax = new Vector2(-14f, 0f);

        TextMeshProUGUI value = (TextMeshProUGUI)AddText(viewport, string.Empty, 22, 50);
        value.alignment = TextAlignmentOptions.MidlineLeft;
        value.color = new Color(.1f, .1f, .1f);
        Stretch(value.rectTransform);
        TextMeshProUGUI placeholder = (TextMeshProUGUI)AddText(viewport, hint, 22, 50);
        placeholder.alignment = TextAlignmentOptions.MidlineLeft;
        placeholder.color = new Color(.5f, .5f, .5f);
        Stretch(placeholder.rectTransform);

        TMP_InputField field = go.GetComponent<TMP_InputField>();
        field.textViewport = viewport;
        field.textComponent = value;
        field.placeholder = placeholder;
        if (password) field.contentType = TMP_InputField.ContentType.Password;
        return field;
    }

    private Button AddButton(Transform parent, string caption, UnityEngine.Events.UnityAction action)
    {
        GameObject go = new(caption, typeof(RectTransform), typeof(Image), typeof(Button), typeof(LayoutElement));
        go.transform.SetParent(parent, false);
        go.GetComponent<Image>().color = new Color(.95f, .72f, .25f);
        go.GetComponent<LayoutElement>().preferredHeight = 45f;
        Button button = go.GetComponent<Button>();
        button.onClick.AddListener(action);
        TMP_Text label = AddText(go.transform, caption, 20, 45);
        label.color = new Color(.1f, .12f, .2f);
        Stretch(label.rectTransform);
        return button;
    }

    private static void Stretch(RectTransform rect)
    {
        rect.anchorMin = Vector2.zero;
        rect.anchorMax = Vector2.one;
        rect.offsetMin = rect.offsetMax = Vector2.zero;
    }
}
