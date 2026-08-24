using System.Collections.Generic;
using SumoBot.Graph;
using UnityEngine;

// Orchestrates the wiring gesture (E3.2, Phase 3). PortView reports drag/drop/right-click here; this
// turns them into GraphDocument.AddConnection / RemoveConnection and manages the ConnectionView
// visuals. Direction (Out -> In) and value-type must match or the wire is refused at the gesture —
// everything else the board permits, leaving fan-in / cycles for GraphValidator to surface (Phase 5).
//
// Lives on EditorRoot so PortView can reach it with GetComponentInParent, and so it can find every
// PortView under the board when drawing a saved graph's existing wires.
public class WiringController : MonoBehaviour
{
    [SerializeField] private GraphEditorController editor;
    [SerializeField] private RectTransform connectionLayer;

    private ConnectionView pending;
    private bool builtExisting;

    // Draw a loaded draft's wires once the nodes/ports exist (after all Start() have run).
    void LateUpdate()
    {
        if (builtExisting || editor == null) return;
        builtExisting = true;
        BuildExistingConnections();
    }

    // ---- gesture entry points (called by PortView) -----------------------------------------

    public void BeginWire(PortView from)
    {
        CancelWire();
        if (from == null) return;
        pending = CreateWire(from, null);
        pending.SetFreeEnd(RectTransformUtility.WorldToScreenPoint(null, from.WorldPosition));
    }

    public void UpdateWire(Vector2 screenPoint)
    {
        if (pending != null) pending.SetFreeEnd(screenPoint);
    }

    public void CancelWire()
    {
        if (pending != null) Destroy(pending.gameObject);
        pending = null;
    }

    public void CompleteWire(PortView target)
    {
        if (pending == null) return;
        PortView source = pending.From;
        CancelWire();

        if (!CanConnect(source, target, out PortView outPort, out PortView inPort)) return;
        if (AlreadyConnected(outPort, inPort)) return;

        GraphConnection conn = editor.Document.AddConnection(outPort.NodeId, outPort.PortId, inPort.NodeId, inPort.PortId);
        if (conn == null) return;

        ConnectionView view = CreateWire(outPort, inPort);
        view.SetConnection(conn.FromNodeId, conn.FromPortId, conn.ToNodeId, conn.ToPortId);
    }

    /// <summary>Right-clicking a port cuts every wire touching it.</summary>
    public void RemovePort(PortView port)
    {
        if (port == null) return;
        foreach (ConnectionView view in connectionLayer.GetComponentsInChildren<ConnectionView>())
        {
            if (view.IsPending) continue;
            bool touches = (view.FromNodeId == port.NodeId && view.FromPortId == port.PortId) ||
                           (view.ToNodeId == port.NodeId && view.ToPortId == port.PortId);
            if (!touches) continue;

            editor.Document.RemoveConnection(view.FromNodeId, view.FromPortId, view.ToNodeId, view.ToPortId);
            Destroy(view.gameObject);
        }
    }

    // ---- rules -----------------------------------------------------------------------------

    // A legal wire joins one Out to one In, of the same value type, on two different nodes.
    private static bool CanConnect(PortView a, PortView b, out PortView outPort, out PortView inPort)
    {
        outPort = inPort = null;
        if (a == null || b == null || a == b || a.NodeId == b.NodeId) return false;

        if (a.Direction == PortDirection.Out && b.Direction == PortDirection.In) { outPort = a; inPort = b; }
        else if (a.Direction == PortDirection.In && b.Direction == PortDirection.Out) { outPort = b; inPort = a; }
        else return false; // same direction — reject

        return outPort.Type == inPort.Type; // type mismatch — reject
    }

    private bool AlreadyConnected(PortView outPort, PortView inPort)
    {
        foreach (GraphConnection c in editor.Document.Connections)
            if (c.FromNodeId == outPort.NodeId && c.FromPortId == outPort.PortId &&
                c.ToNodeId == inPort.NodeId && c.ToPortId == inPort.PortId)
                return true;
        return false;
    }

    // ---- visuals ---------------------------------------------------------------------------

    private ConnectionView CreateWire(PortView from, PortView to)
    {
        var go = new GameObject("Connection", typeof(RectTransform));
        var rt = (RectTransform)go.transform;
        rt.SetParent(connectionLayer, false);
        rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
        rt.pivot = new Vector2(0.5f, 0.5f);
        rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;

        ConnectionView view = go.AddComponent<ConnectionView>();
        view.color = UITheme.Instance.GraphWire;
        view.SetEndpoints(from, to);
        return view;
    }

    private void BuildExistingConnections()
    {
        Dictionary<(string, string, PortDirection), PortView> ports = IndexPorts();
        foreach (GraphConnection c in editor.Document.Connections)
        {
            if (!ports.TryGetValue((c.FromNodeId, c.FromPortId, PortDirection.Out), out PortView from)) continue;
            if (!ports.TryGetValue((c.ToNodeId, c.ToPortId, PortDirection.In), out PortView to)) continue;
            ConnectionView view = CreateWire(from, to);
            view.SetConnection(c.FromNodeId, c.FromPortId, c.ToNodeId, c.ToPortId);
        }
    }

    private Dictionary<(string, string, PortDirection), PortView> IndexPorts()
    {
        var map = new Dictionary<(string, string, PortDirection), PortView>();
        foreach (PortView p in GetComponentsInChildren<PortView>(true))
            map[(p.NodeId, p.PortId, p.Direction)] = p;
        return map;
    }
}
