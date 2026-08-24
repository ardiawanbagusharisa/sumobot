using TMPro;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

// One-click scaffolder for the BotGraphEditor scene (E3.2). It builds the full object tree, generates
// the three palette/node prefabs, and wires every serialized field on GraphEditorController /
// PaletteController / WiringController / ValidationController that points at a scene object or prefab.
//
// Two rules keep the output sane:
//
//  1. Rebuild, don't append. Running this twice used to leave two of everything — including two
//     GraphEditorControllers, i.e. two independent GraphDocuments loading and saving the same draft.
//     Build() now destroys any root object named EditorCanvas first, so a re-run is a rebuild.
//     Nothing outside that root is touched.
//
//  2. Layer by role, not by convenience. Three sibling layers under EditorRoot, in draw order:
//       BoardLayer   — the canvas content: masked viewport > BoardContent > wires + nodes.
//                      Anything that should pan/zoom with the graph belongs under BoardContent.
//       ChromeLayer  — the frame: TopBar, PalettePanel, ValidationPanel. Screen-fixed, never pans.
//       OverlayLayer — pointer-following visuals (the in-progress wire, drag ghosts, tooltips).
//     Panels are laid out from shared constants below, so the board and the palette can't overlap.
//
// Colours come from UITheme (Resources/Theme/UITheme) — the same asset the runtime-built ports, params
// and wires read — so retinting is still "edit one asset". Panel/card shades are derived from the
// theme's Surface role, and text colour is picked for contrast against whatever surface it lands on,
// so the scaffold stays legible under a light or a dark theme.
//
// Usage: open the target scene, then Tools > Sumobot > Build BotGraphEditor Scene. Save as
// Assets/Scenes/BotGraphEditor.unity and add it to Build Settings.
public static class BotGraphEditorSceneBuilder
{
    private const string PrefabDir = "Assets/Prefabs/Workshop";
    private const string CanvasName = "EditorCanvas";

    // ---- Layout constants -----------------------------------------------------------------------
    // Every panel edge is derived from these, so the board can never underlap the palette again.
    private const float TopBarHeight = 96f;
    private const float PaletteWidth = 380f;
    private const float ValidationHeight = 120f;
    private const float Pad = 8f;      // inner padding shared by panels
    private const float Gutter = 24f;  // edge margin for top-bar controls

    // ---- Theme ----------------------------------------------------------------------------------

    private static UITheme Theme => UITheme.Instance;

    private static Color BoardBg => Theme.Background;
    private static Color PanelBg => Theme.Surface;                 // TopBar, ValidationPanel
    private static Color PaletteBg => Elevate(Theme.Surface, 0.05f);
    private static Color CardBg => Elevate(Theme.Surface, 0.10f);  // palette items, node cards
    private static Color FieldBg => Elevate(Theme.Surface, 0.16f); // recessed inputs
    private static readonly Color Transparent = new(0, 0, 0, 0);

    // Perceived luminance (sRGB weights). Used to decide which way "up" is for a surface and which
    // text colour reads against it.
    private static float Luminance(Color c) => 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;

    private static bool IsDark(Color c) => Luminance(c) < 0.5f;

    /// <summary>One step of elevation away from a surface: lighter on a dark theme, darker on a light one.</summary>
    private static Color Elevate(Color c, float amount)
    {
        float target = IsDark(c) ? 1f : 0f;
        return new Color(Mathf.Lerp(c.r, target, amount), Mathf.Lerp(c.g, target, amount), Mathf.Lerp(c.b, target, amount), c.a);
    }

    // Readable text for a background. The theme's TextPrimary is only safe on dark surfaces — the
    // current asset pairs a dark Background with a near-white Surface — so light surfaces get ink
    // derived from themselves instead of a white-on-white panel.
    private static Color TextOn(Color bg) => IsDark(bg) ? Theme.TextPrimary : Elevate(bg, 0.88f);

    private static Color MutedOn(Color bg) => Color.Lerp(TextOn(bg), bg, 0.4f);

    // ---- Build ----------------------------------------------------------------------------------

    [MenuItem("Tools/Sumobot/Build BotGraphEditor Scene")]
    public static void Build()
    {
        int removed = ClearPreviousBuild();
        EnsureEventSystem();

        Canvas canvas = MakeCanvas();
        RectTransform root = NewUI("EditorRoot", canvas.transform);
        Stretch(root, 0, 0, 0, 0);

        GraphEditorController editor = root.gameObject.AddComponent<GraphEditorController>();
        WiringController wiring = root.gameObject.AddComponent<WiringController>();
        ValidationController validation = root.gameObject.AddComponent<ValidationController>();

        // Layers are created in draw order: board content, then chrome over it, then overlay on top.
        BuildBoardLayer(root, out RectTransform connectionLayer, out RectTransform nodesContent, out RectTransform emptyHint);

        RectTransform chrome = NewUI("ChromeLayer", root);
        Stretch(chrome, 0, 0, 0, 0);
        BuildTopBar(chrome, editor);
        PaletteController paletteCtrl = BuildPalettePanel(chrome, out RectTransform paletteContent);
        BuildValidationPanel(chrome, out TMP_Text statusText);

        // Empty for now: WiringController still parents the in-progress wire to ConnectionLayer, which
        // draws it *under* the node cards. Give it an `overlayLayer` field pointing here and pending
        // wires (and later drag ghosts / tooltips) render above everything.
        RectTransform overlay = NewUI("OverlayLayer", root);
        Stretch(overlay, 0, 0, 0, 0);

        // ---- Prefabs + wiring --------------------------------------------------------------------
        EnsureFolder(PrefabDir);
        Button itemPrefab = BuildPaletteItemPrefab();
        TextMeshProUGUI headerPrefab = BuildGroupHeaderPrefab();
        NodeView nodePrefab = BuildNodeCardPrefab();

        SetRef(paletteCtrl, "editor", editor);
        SetRef(paletteCtrl, "paletteContent", paletteContent);
        SetRef(paletteCtrl, "itemPrefab", itemPrefab);
        SetRef(paletteCtrl, "headerPrefab", headerPrefab);
        SetRef(paletteCtrl, "nodePrefab", nodePrefab);
        SetRef(paletteCtrl, "nodeParent", nodesContent);
        SetRef(paletteCtrl, "boardEmptyHint", emptyHint.gameObject);

        SetRef(wiring, "editor", editor);
        SetRef(wiring, "connectionLayer", connectionLayer);

        SetRef(validation, "editor", editor);
        SetRef(validation, "statusText", statusText);

        Selection.activeGameObject = canvas.gameObject;
        EditorSceneManager.MarkSceneDirty(EditorSceneManager.GetActiveScene());
        Debug.Log($"[Sumobot] BotGraphEditor scaffold rebuilt ({removed} previous root(s) replaced). " +
                  "Save the scene as Assets/Scenes/BotGraphEditor.unity and add it to Build Settings.");
    }

    // Remove what an earlier run of this builder left behind, so a re-run replaces instead of doubling.
    // Scoped to root objects named EditorCanvas — everything else in the scene is the author's.
    private static int ClearPreviousBuild()
    {
        int removed = 0;
        foreach (GameObject go in EditorSceneManager.GetActiveScene().GetRootGameObjects())
        {
            if (go.name != CanvasName) continue;
            Object.DestroyImmediate(go);
            removed++;
        }
        return removed;
    }

    private static Canvas MakeCanvas()
    {
        GameObject go = NewUI(CanvasName, null).gameObject;
        Canvas canvas = go.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;

        CanvasScaler scaler = go.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        scaler.matchWidthOrHeight = 1f;

        go.AddComponent<GraphicRaycaster>();
        return canvas;
    }

    // ---- Layers -----------------------------------------------------------------------------------

    // BoardLayer is the viewport: it masks, so a node dragged past the edge is clipped instead of
    // painting over the palette. BoardContent is the pannable space — node coordinates are relative to
    // it, so adding pan/zoom later means moving this one transform and nothing else.
    private static void BuildBoardLayer(RectTransform parent, out RectTransform connectionLayer,
                                        out RectTransform nodesContent, out RectTransform emptyHint)
    {
        RectTransform board = NewUI("BoardLayer", parent);
        AddImage(board, BoardBg);
        board.gameObject.AddComponent<RectMask2D>();
        Stretch(board, PaletteWidth, ValidationHeight, 0, TopBarHeight);

        RectTransform content = NewUI("BoardContent", board);
        Stretch(content, 0, 0, 0, 0);

        connectionLayer = NewUI("ConnectionLayer", content); // first child -> wires draw under nodes
        Stretch(connectionLayer, 0, 0, 0, 0);

        nodesContent = NewUI("NodesContent", content);
        Stretch(nodesContent, 0, 0, 0, 0);

        // Viewport-space, not board-space: the hint should stay centred even once the board pans.
        emptyHint = NewUI("BoardEmptyHint", board);
        TextMeshProUGUI hint = emptyHint.gameObject.AddComponent<TextMeshProUGUI>();
        hint.text = "Pick a module to start";
        hint.alignment = TextAlignmentOptions.Center;
        hint.fontSize = 28;
        hint.color = MutedOn(BoardBg);
        hint.raycastTarget = false; // covers the whole board — must never swallow node clicks
        Stretch(emptyHint, 0, 0, 0, 0);
    }

    private static void BuildTopBar(RectTransform parent, GraphEditorController editor)
    {
        RectTransform topBar = NewUI("TopBar", parent);
        AddImage(topBar, PanelBg);
        AnchorTop(topBar, TopBarHeight);

        RectTransform nameField = MakeInputField("NameField", topBar, "New Bot");
        Anchor(nameField, new Vector2(0, 0.5f), new Vector2(0, 0.5f), new Vector2(0, 0.5f));
        nameField.anchoredPosition = new Vector2(Gutter, 0);
        nameField.sizeDelta = new Vector2(420, 56);

        Button saveBtn = MakeButton("ButtonSave", topBar, "Save", Theme.Primary);
        Button backBtn = MakeButton("ButtonBack", topBar, "Back", CardBg);
        RightAlign((RectTransform)backBtn.transform, -Gutter);
        RightAlign((RectTransform)saveBtn.transform, -(Gutter + ((RectTransform)backBtn.transform).sizeDelta.x + Pad * 2.5f));

        SetRef(editor, "saveButton", saveBtn);
        SetRef(editor, "backButton", backBtn);
        SetRef(editor, "nameField", nameField.GetComponent<TMP_InputField>());
    }

    private static PaletteController BuildPalettePanel(RectTransform parent, out RectTransform content)
    {
        RectTransform palette = NewUI("PalettePanel", parent);
        AddImage(palette, PaletteBg);
        AnchorLeft(palette, PaletteWidth, TopBarHeight);
        PaletteController controller = palette.gameObject.AddComponent<PaletteController>();

        RectTransform header = NewUI("HeaderText", palette);
        TextMeshProUGUI headerText = header.gameObject.AddComponent<TextMeshProUGUI>();
        headerText.text = "Modules";
        headerText.fontSize = 26;
        headerText.color = TextOn(PaletteBg);
        headerText.raycastTarget = false;
        Anchor(header, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 1));
        header.offsetMin = new Vector2(16, -52);
        header.offsetMax = new Vector2(-16, -Pad);

        RectTransform scrollView = NewUI("Scroll View", palette);
        AddImage(scrollView, BoardBg);
        ScrollRect scroll = scrollView.gameObject.AddComponent<ScrollRect>();
        Stretch(scrollView, Pad, Pad, Pad, 56);

        RectTransform viewport = NewUI("Viewport", scrollView);
        AddImage(viewport, Transparent);
        viewport.gameObject.AddComponent<RectMask2D>();
        Stretch(viewport, 0, 0, 0, 0);

        content = NewUI("Content", viewport);
        Anchor(content, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 1));
        content.sizeDelta = Vector2.zero;

        var layout = content.gameObject.AddComponent<VerticalLayoutGroup>();
        layout.childControlHeight = true; layout.childForceExpandHeight = false;
        layout.childControlWidth = true; layout.childForceExpandWidth = true;
        layout.spacing = 4; layout.padding = new RectOffset(6, 6, 6, 6);

        var fitter = content.gameObject.AddComponent<ContentSizeFitter>();
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;

        scroll.viewport = viewport;
        scroll.content = content;
        scroll.horizontal = false;
        scroll.vertical = true;
        return controller;
    }

    // Chrome, not board content: a sibling of BoardLayer so it neither overlaps the node area nor
    // travels with a future board pan.
    private static void BuildValidationPanel(RectTransform parent, out TMP_Text statusText)
    {
        RectTransform panel = NewUI("ValidationPanel", parent);
        AddImage(panel, PanelBg);
        Anchor(panel, new Vector2(0, 0), new Vector2(1, 0), new Vector2(0.5f, 0));
        panel.offsetMin = new Vector2(PaletteWidth, 0);
        panel.offsetMax = new Vector2(0, ValidationHeight);

        RectTransform status = NewUI("StatusText", panel);
        var text = status.gameObject.AddComponent<TextMeshProUGUI>();
        text.text = "Graph is valid";
        text.fontSize = 24;
        text.color = TextOn(PanelBg);
        text.raycastTarget = false;
        Anchor(status, new Vector2(0, 0.5f), new Vector2(1, 0.5f), new Vector2(0.5f, 0.5f));
        status.offsetMin = new Vector2(16, -30);
        status.offsetMax = new Vector2(-16, 30);

        statusText = text;
    }

    // ---- Prefab builders --------------------------------------------------------------------------

    private static Button BuildPaletteItemPrefab()
    {
        GameObject go = new GameObject("PaletteItem", typeof(RectTransform));
        AddImage((RectTransform)go.transform, CardBg);
        Button btn = go.AddComponent<Button>();
        go.AddComponent<LayoutElement>().minHeight = 44;

        RectTransform label = NewUI("Label", go.transform);
        var text = label.gameObject.AddComponent<TextMeshProUGUI>();
        text.text = "Module";
        text.alignment = TextAlignmentOptions.MidlineLeft;
        text.fontSize = 20;
        text.color = TextOn(CardBg);
        text.raycastTarget = false;
        Stretch(label, 12, 0, 10, 0);

        Button prefabBtn = SavePrefab(go, "PaletteItem").GetComponent<Button>();
        Object.DestroyImmediate(go);
        return prefabBtn;
    }

    private static TextMeshProUGUI BuildGroupHeaderPrefab()
    {
        GameObject go = new GameObject("GroupHeader", typeof(RectTransform));
        var text = go.AddComponent<TextMeshProUGUI>();
        text.text = "Group";
        text.fontSize = 18;
        text.fontStyle = FontStyles.UpperCase;
        text.color = MutedOn(PaletteBg);
        text.raycastTarget = false;
        go.AddComponent<LayoutElement>().minHeight = 30;

        var prefabText = SavePrefab(go, "GroupHeader").GetComponent<TextMeshProUGUI>();
        Object.DestroyImmediate(go);
        return prefabText;
    }

    private static NodeView BuildNodeCardPrefab()
    {
        GameObject go = new GameObject("NodeCard", typeof(RectTransform));
        RectTransform rt = (RectTransform)go.transform;
        Anchor(rt, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f));
        rt.sizeDelta = new Vector2(210, 120);
        AddImage(rt, CardBg);
        NodeView view = go.AddComponent<NodeView>();

        RectTransform title = NewUI("Title", go.transform);
        var titleText = title.gameObject.AddComponent<TextMeshProUGUI>();
        titleText.text = "Node";
        titleText.fontSize = 20;
        titleText.color = TextOn(CardBg);
        titleText.raycastTarget = false; // the card handles the drag; the label must not eat it
        Anchor(title, new Vector2(0, 1), new Vector2(1, 1), new Vector2(0.5f, 1));
        title.offsetMin = new Vector2(10, -40);
        title.offsetMax = new Vector2(-10, -Pad);

        // Params (filled at runtime by NodeView): fills everything below the title rather than a fixed
        // slab, so a two-parameter module doesn't spill out of the card.
        RectTransform paramsRT = NewUI("ParamsContainer", go.transform);
        Stretch(paramsRT, Pad, Pad, Pad, 46);
        var paramsLayout = paramsRT.gameObject.AddComponent<VerticalLayoutGroup>();
        paramsLayout.spacing = 4;
        paramsLayout.childControlWidth = true; paramsLayout.childForceExpandWidth = true;
        paramsLayout.childControlHeight = true; paramsLayout.childForceExpandHeight = false;
        paramsLayout.childAlignment = TextAnchor.UpperCenter;

        // Ports: narrow columns pinned to the card's left / right edges, sized by their contents so a
        // module with four inputs doesn't squash them on top of each other.
        MakePortColumn("InputPorts", go.transform, new Vector2(0, 0.5f));
        MakePortColumn("OutputPorts", go.transform, new Vector2(1, 0.5f));

        SetRef(view, "titleLabel", titleText); // rect left null -> resolves to own transform at runtime

        NodeView prefabView = SavePrefab(go, "NodeCard").GetComponent<NodeView>();
        Object.DestroyImmediate(go);
        return prefabView;
    }

    private static void MakePortColumn(string name, Transform parent, Vector2 anchor)
    {
        RectTransform rt = NewUI(name, parent);
        Anchor(rt, anchor, anchor, anchor);
        rt.sizeDelta = new Vector2(24, 40);

        var layout = rt.gameObject.AddComponent<VerticalLayoutGroup>();
        layout.spacing = 4;
        layout.childControlWidth = false; layout.childForceExpandWidth = false;
        layout.childControlHeight = false; layout.childForceExpandHeight = false;
        layout.childAlignment = TextAnchor.MiddleCenter;

        var fitter = rt.gameObject.AddComponent<ContentSizeFitter>();
        fitter.verticalFit = ContentSizeFitter.FitMode.PreferredSize;
    }

    // ---- Small helpers ----------------------------------------------------------------------------

    private static RectTransform NewUI(string name, Transform parent)
    {
        GameObject go = new GameObject(name, typeof(RectTransform));
        if (parent != null) go.transform.SetParent(parent, false);
        return (RectTransform)go.transform;
    }

    private static Image AddImage(RectTransform rt, Color color)
    {
        Image img = rt.gameObject.AddComponent<Image>();
        img.color = color;
        img.raycastTarget = true;
        return img;
    }

    private static Button MakeButton(string name, Transform parent, string label, Color bg)
    {
        RectTransform rt = NewUI(name, parent);
        AddImage(rt, bg);
        Button btn = rt.gameObject.AddComponent<Button>();
        rt.sizeDelta = new Vector2(150, 56);

        RectTransform labelRT = NewUI("Label", rt);
        var text = labelRT.gameObject.AddComponent<TextMeshProUGUI>();
        text.text = label;
        text.alignment = TextAlignmentOptions.Center;
        text.fontSize = 22;
        text.color = TextOn(bg);
        text.raycastTarget = false;
        Stretch(labelRT, 0, 0, 0, 0);
        return btn;
    }

    private static RectTransform MakeInputField(string name, Transform parent, string value)
    {
        RectTransform rt = NewUI(name, parent);
        AddImage(rt, FieldBg);
        TMP_InputField field = rt.gameObject.AddComponent<TMP_InputField>();

        RectTransform area = NewUI("Text Area", rt);
        area.gameObject.AddComponent<RectMask2D>();
        Stretch(area, 10, 6, 10, 6);

        RectTransform textRT = NewUI("Text", area);
        var text = textRT.gameObject.AddComponent<TextMeshProUGUI>();
        text.fontSize = 22;
        text.color = TextOn(FieldBg);
        Stretch(textRT, 0, 0, 0, 0);

        field.textViewport = area;
        field.textComponent = text;
        field.caretColor = TextOn(FieldBg);
        field.customCaretColor = true;
        field.text = value;
        return rt;
    }

    // Anchor with matching min/max (a point anchor).
    private static void Anchor(RectTransform rt, Vector2 min, Vector2 max, Vector2 pivot)
    {
        rt.anchorMin = min; rt.anchorMax = max; rt.pivot = pivot;
        rt.anchoredPosition = Vector2.zero;
    }

    // Full stretch with per-edge insets (left, bottom, right, top).
    private static void Stretch(RectTransform rt, float l, float b, float r, float t)
    {
        rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one; rt.pivot = new Vector2(0.5f, 0.5f);
        rt.offsetMin = new Vector2(l, b);
        rt.offsetMax = new Vector2(-r, -t);
    }

    private static void AnchorTop(RectTransform rt, float height)
    {
        rt.anchorMin = new Vector2(0, 1); rt.anchorMax = new Vector2(1, 1); rt.pivot = new Vector2(0.5f, 1);
        rt.offsetMin = new Vector2(0, -height);
        rt.offsetMax = Vector2.zero;
    }

    private static void AnchorLeft(RectTransform rt, float width, float topInset)
    {
        rt.anchorMin = new Vector2(0, 0); rt.anchorMax = new Vector2(0, 1); rt.pivot = new Vector2(0, 0.5f);
        rt.offsetMin = Vector2.zero;
        rt.offsetMax = new Vector2(width, -topInset);
    }

    private static void RightAlign(RectTransform rt, float x)
    {
        rt.anchorMin = new Vector2(1, 0.5f); rt.anchorMax = new Vector2(1, 0.5f); rt.pivot = new Vector2(1, 0.5f);
        rt.anchoredPosition = new Vector2(x, 0);
    }

    private static void EnsureEventSystem()
    {
        EventSystem[] existing = Object.FindObjectsByType<EventSystem>(FindObjectsSortMode.None);
        if (existing.Length == 1) return;
        if (existing.Length > 1)
        {
            Debug.LogWarning($"[Sumobot] {existing.Length} EventSystems in this scene — input behaviour is undefined. Delete the extras.");
            return;
        }

        GameObject es = new GameObject("EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
        Undo.RegisterCreatedObjectUndo(es, "Create EventSystem");
    }

    private static void EnsureFolder(string path)
    {
        if (AssetDatabase.IsValidFolder(path)) return;
        string parent = System.IO.Path.GetDirectoryName(path).Replace('\\', '/');
        if (!AssetDatabase.IsValidFolder(parent)) EnsureFolder(parent);
        AssetDatabase.CreateFolder(parent, System.IO.Path.GetFileName(path));
    }

    private static GameObject SavePrefab(GameObject go, string name)
        => PrefabUtility.SaveAsPrefabAsset(go, $"{PrefabDir}/{name}.prefab");

    // Assign a private [SerializeField] reference via SerializedObject (works for private fields).
    private static void SetRef(Object target, string field, Object value)
    {
        var so = new SerializedObject(target);
        var prop = so.FindProperty(field);
        if (prop == null) { Debug.LogWarning($"[Sumobot] No serialized field '{field}' on {target.GetType().Name}."); return; }
        prop.objectReferenceValue = value;
        so.ApplyModifiedPropertiesWithoutUndo();
    }
}
