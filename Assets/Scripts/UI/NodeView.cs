using SumoBot.Graph;
using SumoBot.Graph.Authoring;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

// The on-board visual for one graph node (E3.2). Phase 1 showed the name; Phase 2 renders its ports
// from the ModuleDefinition and makes the card draggable (-> Document.MoveNode). It stays thin and
// keyed by NodeId: the model is authoritative, so a drag just reports the new position and a rebind
// re-reads everything. Port containers are found by name convention (InputPorts / OutputPorts on the
// NodeCard prefab) so no extra Inspector wiring is needed. Phase 3 adds wiring on the ports; Phase 5
// adds error highlighting here.
public class NodeView : MonoBehaviour, IBeginDragHandler, IDragHandler
{
    [SerializeField] private TMP_Text titleLabel;
    [SerializeField] private RectTransform rect; // this card's transform; defaults to own RectTransform

    /// <summary>The backing node's id — the only model state a view carries.</summary>
    public string NodeId { get; private set; }

    private GraphDocument document;
    private RectTransform parentRect; // NodesContent — the space drags are measured in
    private Vector2 grabOffset;

    /// <summary>Bind this card to a node: title, position, and a port per ModuleDefinition port.</summary>
    public void Bind(GraphNode node, ModuleDefinition def, GraphDocument doc)
    {
        NodeId = node.NodeId;
        document = doc;
        if (rect == null) rect = (RectTransform)transform;
        parentRect = rect.parent as RectTransform;
        rect.anchoredPosition = new Vector2(node.X, node.Y);
        if (titleLabel != null) titleLabel.text = def != null ? def.DisplayName : node.TypeId;
        BuildPorts(def);
        BuildParams(def, node);
    }

    private void BuildPorts(ModuleDefinition def)
    {
        RectTransform inputs = transform.Find("InputPorts") as RectTransform;
        RectTransform outputs = transform.Find("OutputPorts") as RectTransform;
        ClearChildren(inputs);
        ClearChildren(outputs);
        if (def == null) return;

        if (inputs != null) foreach (PortSpec p in def.Inputs) CreatePort(p, inputs);
        if (outputs != null) foreach (PortSpec p in def.Outputs) CreatePort(p, outputs);
    }

    // A port = a small dot (Image + PortView) with a label placed on the inside of the card.
    private void CreatePort(PortSpec spec, RectTransform container)
    {
        var portGO = new GameObject($"Port_{spec.Id}", typeof(RectTransform));
        var portRect = (RectTransform)portGO.transform;
        portRect.SetParent(container, false);
        portRect.sizeDelta = new Vector2(18, 18);
        var element = portGO.AddComponent<LayoutElement>();
        element.minWidth = 18; element.minHeight = 18;

        Image dot = portGO.AddComponent<Image>();
        PortView view = portGO.AddComponent<PortView>();

        bool isInput = spec.Direction == PortDirection.In;
        var labelGO = new GameObject("Label", typeof(RectTransform));
        var labelRect = (RectTransform)labelGO.transform;
        labelRect.SetParent(portRect, false);
        labelRect.anchorMin = labelRect.anchorMax = new Vector2(0.5f, 0.5f);
        labelRect.pivot = new Vector2(isInput ? 0f : 1f, 0.5f);
        labelRect.sizeDelta = new Vector2(96, 18);
        labelRect.anchoredPosition = new Vector2(isInput ? 14 : -14, 0);
        var label = labelGO.AddComponent<TextMeshProUGUI>();
        label.fontSize = 14;
        label.color = UITheme.Instance.TextPrimary;
        label.alignment = isInput ? TextAlignmentOptions.MidlineLeft : TextAlignmentOptions.MidlineRight;

        view.Bind(NodeId, spec, dot, label);
    }

    // Params (Phase 4): a numeric field per ParamSpec, dropped into the card's ParamsContainer.
    private void BuildParams(ModuleDefinition def, GraphNode node)
    {
        RectTransform container = transform.Find("ParamsContainer") as RectTransform;
        ClearChildren(container);
        if (container == null || def == null) return;

        foreach (ParamSpec spec in def.Parameters)
        {
            float current = node.Params.TryGetValue(spec.Id, out float v) ? v : spec.Default;
            CreateParamField(spec, current, container);
        }
    }

    private void CreateParamField(ParamSpec spec, float current, RectTransform container)
    {
        var rowGO = new GameObject($"Param_{spec.Id}", typeof(RectTransform));
        var row = (RectTransform)rowGO.transform;
        row.SetParent(container, false);
        var hlg = rowGO.AddComponent<HorizontalLayoutGroup>();
        hlg.spacing = 6;
        hlg.childControlWidth = true; hlg.childForceExpandWidth = false;
        hlg.childControlHeight = true; hlg.childForceExpandHeight = true;
        rowGO.AddComponent<LayoutElement>().minHeight = 28;

        var nameGO = new GameObject("Name", typeof(RectTransform));
        nameGO.transform.SetParent(row, false);
        var nameLabel = nameGO.AddComponent<TextMeshProUGUI>();
        nameLabel.fontSize = 16;
        nameLabel.color = UITheme.Instance.TextPrimary;
        nameLabel.alignment = TextAlignmentOptions.MidlineLeft;
        var nameLE = nameGO.AddComponent<LayoutElement>();
        nameLE.minWidth = 70; nameLE.flexibleWidth = 1;

        TMP_InputField field = BuildInputField(row);
        var view = rowGO.AddComponent<ParamFieldView>();
        view.Bind(NodeId, spec, current, document, field, nameLabel);
    }

    private static TMP_InputField BuildInputField(RectTransform parent)
    {
        var go = new GameObject("Value", typeof(RectTransform));
        var rt = (RectTransform)go.transform;
        rt.SetParent(parent, false);
        go.AddComponent<Image>().color = UITheme.Instance.Surface;
        TMP_InputField field = go.AddComponent<TMP_InputField>();
        var le = go.AddComponent<LayoutElement>();
        le.minWidth = 72; le.preferredWidth = 84;

        var area = new GameObject("Text Area", typeof(RectTransform));
        var areaRT = (RectTransform)area.transform;
        areaRT.SetParent(rt, false);
        area.AddComponent<RectMask2D>();
        StretchLocal(areaRT, 6, 3, 6, 3);

        var textGO = new GameObject("Text", typeof(RectTransform));
        var textRT = (RectTransform)textGO.transform;
        textRT.SetParent(areaRT, false);
        var text = textGO.AddComponent<TextMeshProUGUI>();
        text.fontSize = 16;
        text.color = UITheme.Instance.TextPrimary;
        StretchLocal(textRT, 0, 0, 0, 0);

        field.textViewport = areaRT;
        field.textComponent = text;
        return field;
    }

    private static void StretchLocal(RectTransform rt, float l, float b, float r, float t)
    {
        rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one; rt.pivot = new Vector2(0.5f, 0.5f);
        rt.offsetMin = new Vector2(l, b);
        rt.offsetMax = new Vector2(-r, -t);
    }

    // Error highlight (Phase 5): toggled by ValidationController for offending nodes.
    private Outline errorOutline;

    public void SetErrorHighlight(bool on)
    {
        if (errorOutline == null)
        {
            errorOutline = GetComponent<Outline>();
            if (errorOutline == null)
            {
                errorOutline = gameObject.AddComponent<Outline>();
                errorOutline.effectColor = UITheme.Instance.Danger;
                errorOutline.effectDistance = new Vector2(3, 3);
            }
        }
        errorOutline.enabled = on;
    }

    private static void ClearChildren(Transform t)
    {
        if (t == null) return;
        for (int i = t.childCount - 1; i >= 0; i--)
            Destroy(t.GetChild(i).gameObject);
    }

    public void OnBeginDrag(PointerEventData eventData)
    {
        if (parentRect == null) return;
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(parentRect, eventData.position, eventData.pressEventCamera, out Vector2 local))
            grabOffset = rect.anchoredPosition - local;
    }

    public void OnDrag(PointerEventData eventData)
    {
        if (parentRect == null) return;
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(parentRect, eventData.position, eventData.pressEventCamera, out Vector2 local))
            return;

        rect.anchoredPosition = local + grabOffset;
        document?.MoveNode(NodeId, rect.anchoredPosition.x, rect.anchoredPosition.y);
    }
}
