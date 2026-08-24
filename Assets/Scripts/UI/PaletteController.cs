using System.Collections.Generic;
using SumoBot.Graph;
using SumoBot.Graph.Authoring;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Phase 1 of the node editor (E3.2, task 21.3.2 AC#1/#5). Builds the module palette from the
// GraphEditorController's ModuleLibrary — grouped by ModuleKind (Sensor / Logic / Action), so it
// auto-syncs with the vocabulary — and turns a click into a placed node: Document.AddNode(...) plus
// a NodeView on the board. Any pre-existing nodes (editing a saved draft) are rendered on load.
//
// It only ever calls the authoritative model (Document.AddNode) and reads Document.Nodes; it holds
// no graph state of its own. Board node management lives here for now; Phase 2 may extract a
// dedicated board view as node interaction grows.
public class PaletteController : MonoBehaviour
{
    [SerializeField] private GraphEditorController editor; // source of the live Document + Library

    [Header("Palette (left panel)")]
    [SerializeField] private RectTransform paletteContent; // Scroll View/Viewport/Content (Vertical Layout Group)
    [SerializeField] private Button itemPrefab;            // a button with a child TMP label
    [SerializeField] private TMP_Text headerPrefab;        // a group header label (Sensor/Logic/Action)

    [Header("Board")]
    [SerializeField] private NodeView nodePrefab;          // the on-board node card
    [SerializeField] private RectTransform nodeParent;     // Board/NodesContent — spawned nodes go here
    [SerializeField] private GameObject boardEmptyHint;    // shown while the board has no nodes (AC#5)

    // Kinds in palette order (matches the enum, but pinned here so the UI order is intentional).
    private static readonly ModuleKind[] KindOrder = { ModuleKind.Sensor, ModuleKind.Logic, ModuleKind.Action };

    // Where the next picked node lands, cascaded so several picks don't stack exactly on top.
    private int spawnCount;

    void Start()
    {
        if (editor == null)
        {
            Logger.Error("[Workshop] PaletteController has no GraphEditorController assigned.");
            return;
        }

        BuildPalette();
        RenderExistingNodes();
        RefreshEmptyHint();
    }

    // One header + one button per module, grouped by kind. Rebuilt from the library each time so it
    // always reflects the current vocabulary.
    private void BuildPalette()
    {
        if (paletteContent == null || itemPrefab == null) return;

        for (int i = paletteContent.childCount - 1; i >= 0; i--)
            Destroy(paletteContent.GetChild(i).gameObject);

        foreach (ModuleKind kind in KindOrder)
        {
            List<ModuleDefinition> defs = DefsOfKind(kind);
            if (defs.Count == 0) continue;

            if (headerPrefab != null)
            {
                TMP_Text header = Instantiate(headerPrefab, paletteContent);
                header.text = kind.ToString();
            }

            foreach (ModuleDefinition def in defs)
            {
                Button item = Instantiate(itemPrefab, paletteContent);
                TMP_Text label = item.GetComponentInChildren<TMP_Text>();
                if (label != null) label.text = def.DisplayName;

                string typeId = def.TypeId; // capture per iteration
                item.onClick.AddListener(() => Pick(typeId));
            }
        }
    }

    private List<ModuleDefinition> DefsOfKind(ModuleKind kind)
    {
        var list = new List<ModuleDefinition>();
        foreach (ModuleDefinition def in editor.Library.All)
            if (def.Kind == kind)
                list.Add(def);
        return list;
    }

    // Place a new node of this type at the next cascaded position and show it.
    private void Pick(string typeId)
    {
        Vector2 pos = NextSpawnPosition();
        GraphNode node = editor.Document.AddNode(typeId, pos.x, pos.y);
        if (node == null) return; // unknown type — AddNode already guards the safe vocabulary

        SpawnNodeView(node);
        RefreshEmptyHint();
    }

    private void RenderExistingNodes()
    {
        foreach (GraphNode node in editor.Document.Nodes)
            SpawnNodeView(node);
    }

    private void SpawnNodeView(GraphNode node)
    {
        if (nodePrefab == null || nodeParent == null) return;
        NodeView view = Instantiate(nodePrefab, nodeParent);
        view.Bind(node, editor.Library.Get(node.TypeId), editor.Document);
    }

    // Cascade around board center so repeated picks are all visible.
    private Vector2 NextSpawnPosition()
    {
        const float step = 36f;
        int slot = spawnCount++ % 6;
        return new Vector2(slot * step, -slot * step);
    }

    private void RefreshEmptyHint()
    {
        if (boardEmptyHint != null)
            boardEmptyHint.SetActive(editor.Document.Nodes.Count == 0);
    }
}
