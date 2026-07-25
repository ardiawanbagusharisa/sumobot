using System.Threading.Tasks;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Account panel (MenuAccount): a standalone, always-reachable screen for the local guest
// identity — view PlayerId, see the Guest badge, and edit the display name at any time
// (previously only possible during the one-time login flow). Backed entirely by IAuthService.
// Sign out intentionally stays in Settings, and account linking/email upgrade is out of scope.
//
// Widgets are found by name under the panel root, so no per-widget Inspector wiring is needed.
// Expected tree (MenuAccount):
//   Header/ButtonExit                  (Button — Back)
//   Panel/DisplayName/**InputField**   (TMP_InputField — the only one in the group)
//   Panel/DisplayName/**BtnSave**      (Button)
//   Panel/PlayerID/**Content**         (TMP_Text, read-only)
//   StatusText                         (TMP_Text, optional — save feedback; not present yet)
public class AccountController : MonoBehaviour
{
    [Tooltip("Optional — auto-found in the scene if left empty.")]
    [SerializeField] private MainMenuController mainMenu;

    [Tooltip("Panel root to bind widgets under. Optional — if empty it is auto-resolved, so " +
             "this component may sit on the panel or on a manager child of it.")]
    [SerializeField] private Transform panelRoot;

    private Transform root;

    private TMP_InputField displayNameInput;
    private Button saveNameButton;
    private TMP_Text playerIdValue;
    private TMP_Text statusText;
    private Button backButton;

    private bool busy;
    private bool bound;

    void Awake() => Bind();

    void OnEnable()
    {
        Bind();
        RefreshFromAccount();
    }

    private void Bind()
    {
        if (bound) return;

        if (mainMenu == null) mainMenu = FindFirstObjectByType<MainMenuController>(FindObjectsInactive.Include);
        root = ResolveRoot();

        // The input is the only TMP_InputField under the DisplayName group, and BtnSave lives
        // there too — scope the button lookup to the group so it can't stray elsewhere.
        var displayNameGroup = FindDeep(root, "DisplayName");
        displayNameInput = ComponentIn<TMP_InputField>(displayNameGroup);
        saveNameButton = ComponentIn<Button>(FindDeep(displayNameGroup, "BtnSave"));

        // PlayerID group holds both a "Label" and the value "Content"; target Content directly
        // so we don't grab the label.
        playerIdValue = ComponentIn<TMP_Text>(FindDeep(FindDeep(root, "PlayerID"), "Content"));

        statusText = ComponentIn<TMP_Text>(FindDeep(root, "StatusText"));
        backButton = ComponentIn<Button>(FindDeep(FindDeep(root, "Header"), "ButtonExit"));

        WireEvents();
        bound = true;
    }

    private void WireEvents()
    {
        if (saveNameButton != null) saveNameButton.onClick.AddListener(() => _ = SaveNameAsync());
        // Submitting the field (Enter / focus loss) saves too, so a mouse click on Save isn't required.
        if (displayNameInput != null) displayNameInput.onSubmit.AddListener(value => _ = SaveNameAsync());
        if (backButton != null) backButton.onClick.AddListener(Back);
    }

    // Pull the signed-in account into the widgets. Called every time the panel opens so it
    // always reflects the current identity (e.g. after a name change elsewhere).
    private void RefreshFromAccount()
    {
        var account = GameServices.Auth?.Current;

        if (displayNameInput != null) displayNameInput.SetTextWithoutNotify(account?.DisplayName ?? string.Empty);
        if (playerIdValue != null) playerIdValue.text = account?.PlayerId ?? "—";
        SetStatus(string.Empty);
    }

    private async Task SaveNameAsync()
    {
        if (busy) return;

        var account = GameServices.Auth?.Current;
        if (account == null)
        {
            SetStatus("Not signed in.");
            return;
        }

        string newName = displayNameInput != null ? displayNameInput.text.Trim() : null;

        // No change — nothing to persist. (Auth would also reject an empty name below.)
        if (string.IsNullOrEmpty(newName) || newName == account.DisplayName)
        {
            RefreshFromAccount(); // normalize the field back to the saved value
            return;
        }

        busy = true;
        var result = await GameServices.Auth.SetDisplayNameAsync(newName);
        busy = false;

        if (!result.Success)
        {
            SetStatus(result.Error);
            RefreshFromAccount(); // never leave a rejected value in the field
            return;
        }

        // Keep the local Left profile / leaderboard display name in sync. PlayerId is unchanged,
        // so ApplyAccount does not trigger a leaderboard profile reassignment.
        GameManager.Instance.ApplyAccount(account.PlayerId, account.DisplayName);

        RefreshFromAccount();
        SetStatus("Saved");
    }

    private void SetStatus(string message)
    {
        if (statusText != null) statusText.text = message;
    }

    private void Back()
    {
        if (mainMenu != null) mainMenu.ShowMainMenu();
    }

    // ---------- Helpers (mirrors SettingsController) ----------

    // Panel to search under: an assigned panelRoot, else the nearest "MenuAccount" ancestor,
    // else any "MenuAccount" in the scene, else self.
    private Transform ResolveRoot()
    {
        if (panelRoot != null) return panelRoot;
        for (Transform t = transform; t != null; t = t.parent)
            if (t.name == PanelName) return t;
        return FindInScene(PanelName) ?? transform;
    }

    private const string PanelName = "MenuAccount";

    // By-name lookup that also sees inactive objects (unlike GameObject.Find).
    private static Transform FindInScene(string objectName)
    {
        foreach (var t in Resources.FindObjectsOfTypeAll<Transform>())
            if (t.name == objectName && t.gameObject.scene.IsValid())
                return t;
        return null;
    }

    private static T ComponentIn<T>(Transform t) where T : Component
        => t != null ? t.GetComponentInChildren<T>(true) : null;

    private static Transform FindDeep(Transform parent, string objectName)
    {
        if (parent == null) return null;
        foreach (Transform child in parent)
        {
            if (child.name == objectName) return child;
            var found = FindDeep(child, objectName);
            if (found != null) return found;
        }
        return null;
    }
}
