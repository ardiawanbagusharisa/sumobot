using System.Collections.Generic;
using System.Threading.Tasks;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

// Settings panel (MenuSetting): Music/SFX volume + mute, language, reset progress, and
// sign out / quit / back. Widgets are found by name under the panel root, so no per-widget
// Inspector wiring is needed. Expected tree:
//   Header/ButtonExit                         (Back)
//   Panel/Audio/{MusicRow,SFXRow}/{Slider,Value,ButtonMuted}
//   Panel/Game/LanguageRow/Dropdown
//   Panel/Data/ResetRow/BtnReset              (opens confirm dialog)
//   Panel/{BtnSignout,BtnQuit}
//   PanelConfirmReset/{BtnCancel,BtnReset}
public class SettingsController : MonoBehaviour
{
    [Tooltip("Optional — auto-found in the scene if left empty.")]
    [SerializeField] private MainMenuController mainMenu;

    [Tooltip("Panel root to bind widgets under. Optional — if empty it is auto-resolved, so " +
             "this component may sit on the panel or on a manager child of it.")]
    [SerializeField] private Transform panelRoot;

    private Transform root;

    private Slider musicSlider;
    private Slider sfxSlider;
    private TMP_Text musicValue;
    private TMP_Text sfxValue;
    private Button musicMuteButton;
    private Button sfxMuteButton;
    private TMP_Text musicMuteLabel;
    private TMP_Text sfxMuteLabel;

    private TMP_Dropdown languageDropdown;
    private Button resetButton;
    private Button signOutButton;
    private Button quitButton;
    private Button backButton;

    private GameObject confirmResetDialog;
    private Button confirmResetCancelButton;
    private Button confirmResetConfirmButton;

    private bool busy;
    private bool bound;

    void Awake() => Bind();

    void OnEnable()
    {
        Bind();
        HideResetConfirm();
        RefreshFromSettings();
    }

    private void Bind()
    {
        if (bound) return;

        if (mainMenu == null) mainMenu = FindFirstObjectByType<MainMenuController>(FindObjectsInactive.Include);
        root = ResolveRoot();

        // Scope each lookup to its row: "Slider"/"Value"/"ButtonMuted" repeat across both rows.
        var musicRow = FindDeep(root, "MusicRow");
        var sfxRow = FindDeep(root, "SFXRow");

        musicSlider = ComponentIn<Slider>(musicRow);
        sfxSlider = ComponentIn<Slider>(sfxRow);
        musicValue = ComponentIn<TMP_Text>(FindDeep(musicRow, "Value"));
        sfxValue = ComponentIn<TMP_Text>(FindDeep(sfxRow, "Value"));
        musicMuteButton = ComponentIn<Button>(FindDeep(musicRow, "ButtonMuted"));
        sfxMuteButton = ComponentIn<Button>(FindDeep(sfxRow, "ButtonMuted"));
        if (musicMuteButton != null) musicMuteLabel = musicMuteButton.GetComponentInChildren<TMP_Text>(true);
        if (sfxMuteButton != null) sfxMuteLabel = sfxMuteButton.GetComponentInChildren<TMP_Text>(true);

        languageDropdown = ComponentIn<TMP_Dropdown>(FindDeep(root, "LanguageRow"));
        resetButton = ComponentIn<Button>(FindDeep(root, "ResetRow"));

        // No dedicated Back button — the header's close (ButtonExit) is it.
        signOutButton = ComponentIn<Button>(FindDeep(root, "BtnSignout"));
        quitButton = ComponentIn<Button>(FindDeep(root, "BtnQuit"));
        backButton = ComponentIn<Button>(FindDeep(FindDeep(root, "Header"), "ButtonExit"));

        // Scope the dialog's buttons inside it — "BtnReset" also exists in ResetRow.
        var dialog = FindDeep(root, "PanelConfirmReset");
        confirmResetDialog = dialog != null ? dialog.gameObject : null;
        confirmResetCancelButton = ComponentIn<Button>(FindDeep(dialog, "BtnCancel"));
        confirmResetConfirmButton = ComponentIn<Button>(FindDeep(dialog, "BtnReset"));

        WireEvents();
        bound = true;
    }

    private void WireEvents()
    {
        if (musicSlider != null) musicSlider.onValueChanged.AddListener(_ => OnMusicChanged());
        if (sfxSlider != null) sfxSlider.onValueChanged.AddListener(_ => OnSfxChanged());
        if (musicMuteButton != null) musicMuteButton.onClick.AddListener(ToggleMusicMute);
        if (sfxMuteButton != null) sfxMuteButton.onClick.AddListener(ToggleSfxMute);
        if (languageDropdown != null) languageDropdown.onValueChanged.AddListener(OnLanguageChanged);
        if (resetButton != null) resetButton.onClick.AddListener(ShowResetConfirm);
        if (confirmResetCancelButton != null) confirmResetCancelButton.onClick.AddListener(HideResetConfirm);
        if (confirmResetConfirmButton != null) confirmResetConfirmButton.onClick.AddListener(() => _ = ResetProgressAsync());
        if (signOutButton != null) signOutButton.onClick.AddListener(() => _ = SignOutAsync());
        if (quitButton != null) quitButton.onClick.AddListener(QuitGame);
        if (backButton != null) backButton.onClick.AddListener(Back);
    }

    // Pull persisted settings into the widgets without re-firing their change handlers.
    private void RefreshFromSettings()
    {
        SetSliderNorm(musicSlider, GameSettings.MusicVolume);
        SetSliderNorm(sfxSlider, GameSettings.SfxVolume);

        SetupLanguageDropdown();

        UpdateMusicLabel();
        UpdateSfxLabel();
        UpdateMuteLabels();

        ApplyMusicVolume();
        ApplySfxVolume();
    }

    private void SetupLanguageDropdown()
    {
        if (languageDropdown == null) return;
        languageDropdown.ClearOptions();
        languageDropdown.AddOptions(new List<string>(GameSettings.SupportedLanguages));

        int idx = System.Array.IndexOf(GameSettings.SupportedLanguages, GameSettings.Language);
        if (idx < 0) idx = 0;
        languageDropdown.SetValueWithoutNotify(idx);
        languageDropdown.RefreshShownValue();
    }

    // ---------- Audio ----------

    private void OnMusicChanged()
    {
        float v = SliderNorm(musicSlider);
        GameSettings.MusicVolume = v;
        if (GameSettings.MusicMuted && v > 0f) { GameSettings.MusicMuted = false; UpdateMuteLabels(); } // dragging up un-mutes
        UpdateMusicLabel();
        ApplyMusicVolume();
    }

    private void OnSfxChanged()
    {
        float v = SliderNorm(sfxSlider);
        GameSettings.SfxVolume = v;
        if (GameSettings.SfxMuted && v > 0f) { GameSettings.SfxMuted = false; UpdateMuteLabels(); }
        UpdateSfxLabel();
        ApplySfxVolume();
    }

    private void ToggleMusicMute()
    {
        GameSettings.MusicMuted = !GameSettings.MusicMuted;
        UpdateMuteLabels();
        ApplyMusicVolume();
    }

    private void ToggleSfxMute()
    {
        GameSettings.SfxMuted = !GameSettings.SfxMuted;
        UpdateMuteLabels();
        ApplySfxVolume();
        if (!GameSettings.SfxMuted && SFXManager.Instance != null) SFXManager.Instance.Play2D("ui_accept"); // cue on un-mute
    }

    private void ApplyMusicVolume()
    {
        if (BGMManager.Instance != null) BGMManager.Instance.SetVolume(GameSettings.EffectiveMusicVolume);
    }

    private void ApplySfxVolume()
    {
        if (SFXManager.Instance != null) SFXManager.Instance.SetMasterVolume(GameSettings.EffectiveSfxVolume);
    }

    private void UpdateMusicLabel()
    {
        if (musicValue != null) musicValue.text = Mathf.RoundToInt(SliderNorm(musicSlider) * 100f) + "%";
    }

    private void UpdateSfxLabel()
    {
        if (sfxValue != null) sfxValue.text = Mathf.RoundToInt(SliderNorm(sfxSlider) * 100f) + "%";
    }

    private void UpdateMuteLabels()
    {
        if (musicMuteLabel != null) musicMuteLabel.text = GameSettings.MusicMuted ? "Unmute" : "Mute";
        if (sfxMuteLabel != null) sfxMuteLabel.text = GameSettings.SfxMuted ? "Unmute" : "Mute";
    }

    // ---------- Language ----------

    private void OnLanguageChanged(int index)
    {
        if (languageDropdown != null && index >= 0 && index < languageDropdown.options.Count)
            GameSettings.Language = languageDropdown.options[index].text;
    }

    // ---------- Reset progress ----------

    private void ShowResetConfirm()
    {
        if (confirmResetDialog != null) confirmResetDialog.SetActive(true);
    }

    private void HideResetConfirm()
    {
        if (confirmResetDialog != null) confirmResetDialog.SetActive(false);
    }

    private async Task ResetProgressAsync()
    {
        if (busy) return;
        busy = true;

        var result = await GameServices.ResetPlayerProgressAsync();
        if (!result.Success)
            Debug.LogError($"[Settings] Reset progress failed: {result.Error}");

        HideResetConfirm();
        busy = false;
    }

    // ---------- Navigation ----------

    private async Task SignOutAsync()
    {
        if (busy) return;
        busy = true;

        await GameServices.Auth.SignOutAsync();
        busy = false;
        // Reload so LoginController runs its sign-in flow again from a clean state.
        SceneManager.LoadScene("MainMenu");
    }

    private void Back()
    {
        if (mainMenu != null) mainMenu.ShowMainMenu();
    }

    private void QuitGame()
    {
        if (mainMenu != null) mainMenu.QuitGame();
        else Application.Quit();
    }

    // ---------- Helpers ----------

    // Normalize to/from 0..1 so the sliders can be authored as either 0..1 or 0..100.
    private static float SliderNorm(Slider s)
    {
        if (s == null || Mathf.Approximately(s.maxValue, s.minValue)) return 0f;
        return Mathf.InverseLerp(s.minValue, s.maxValue, s.value);
    }

    private static void SetSliderNorm(Slider s, float norm01)
    {
        if (s == null) return;
        s.SetValueWithoutNotify(Mathf.Lerp(s.minValue, s.maxValue, Mathf.Clamp01(norm01)));
    }

    // Panel to search under: an assigned panelRoot, else the nearest "MenuSetting" ancestor,
    // else any "MenuSetting" in the scene, else self.
    private Transform ResolveRoot()
    {
        if (panelRoot != null) return panelRoot;
        for (Transform t = transform; t != null; t = t.parent)
            if (t.name == PanelName) return t;
        return FindInScene(PanelName) ?? transform;
    }

    private const string PanelName = "MenuSetting";

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
