using UnityEngine;

// Single source of truth for the game's UI color tone (color base). Named roles instead of literals
// scattered per-controller, so retinting is "edit one asset" instead of hunting down Color(...) calls.
// Loaded from Resources so code-built UI (e.g. the bot-graph board) can read it without scene wiring.
[CreateAssetMenu(fileName = "UITheme", menuName = "Sumobot/UI Theme")]
public class UITheme : ScriptableObject
{
    private const string ResourcePath = "Theme/UITheme";

    // Values sourced from the actual MainMenu.unity design (Image/Text m_Color), not invented placeholders.
    [Header("Core roles")]
    public Color Primary = new(0.49f, 0.59f, 1.00f);       // Background / ButtonAsk / ButtonPublish
    public Color Secondary = new(0.27f, 0.27f, 0.27f);     // BtnSignout / BtnQuit
    public Color Background = new(0.21f, 0.21f, 0.21f);    // Background (dark variant)
    public Color Surface = new(0.96f, 0.96f, 0.96f);       // Item Background / PanelInventory / PanelChat
    public Color TextPrimary = new(1.00f, 1.00f, 1.00f);   // overwhelming majority of MainMenu text
    public Color TextSecondary = new(0.70f, 0.70f, 0.70f); // InputDisplayName
    public Color Danger = new(0.89f, 0.00f, 0.22f);        // ButtonDel
    public Color Success = new(0.00f, 0.70f, 0.17f);       // BtnNewCreation
    public Color Accent = new(0.95f, 0.68f, 0.00f);        // Price / Button_HideInventory / Button_HideChat

    // No direct MainMenu equivalent (wires/ports are graph-only) — reuse core roles to stay in the same palette.
    [Header("Bot graph editor")]
    public Color GraphWire = new(0.27f, 0.27f, 0.27f);     // = Secondary
    public Color PortNumber = new(0.95f, 0.68f, 0.00f);    // = Accent
    public Color PortBool = new(0.49f, 0.59f, 1.00f);      // = Primary

    private static UITheme instance;

    /// <summary>The active palette. Falls back to an in-memory default if the asset is missing so callers never null-check.</summary>
    public static UITheme Instance
    {
        get
        {
            if (instance == null)
            {
                instance = Resources.Load<UITheme>(ResourcePath);
                if (instance == null)
                {
                    Debug.LogWarning($"UITheme asset not found at Resources/{ResourcePath}; using built-in defaults.");
                    instance = CreateInstance<UITheme>();
                }
            }
            return instance;
        }
    }
}
