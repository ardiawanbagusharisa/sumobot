using System.Collections.Generic;
using SumoBot.Graph;
using TMPro;
using UnityEngine;

// Live validation for the editor (E3.2, Phase 5). It listens for structural changes on the
// GraphDocument, re-runs GraphValidator, writes a summary into the ValidationPanel, and highlights
// each offending node (GraphError.NodeId). Runs the pass in LateUpdate so the node views spawned by
// this frame's edit already exist before we try to highlight them.
//
// Lives on EditorRoot so it can find every NodeView under the board.
public class ValidationController : MonoBehaviour
{
    [SerializeField] private GraphEditorController editor;
    [SerializeField] private TMP_Text statusText;

    private bool dirty = true; // validate once on load

    void Start()
    {
        if (editor?.Document != null) editor.Document.Changed += MarkDirty;
    }

    void OnDestroy()
    {
        if (editor?.Document != null) editor.Document.Changed -= MarkDirty;
    }

    private void MarkDirty() => dirty = true;

    void LateUpdate()
    {
        if (!dirty || editor?.Document == null) return;
        dirty = false;
        Revalidate();
    }

    private void Revalidate()
    {
        IReadOnlyList<GraphError> errors = editor.Document.Validate();

        if (statusText != null)
            statusText.text = errors.Count == 0
                ? "Graph is valid"
                : $"{errors.Count} issue(s) — {errors[0].Message}";

        var offending = new HashSet<string>();
        foreach (GraphError error in errors)
            if (!string.IsNullOrEmpty(error.NodeId)) offending.Add(error.NodeId);

        foreach (NodeView node in GetComponentsInChildren<NodeView>(true))
            node.SetErrorHighlight(offending.Contains(node.NodeId));
    }
}
