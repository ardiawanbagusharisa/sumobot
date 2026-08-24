using UnityEngine;
using UnityEngine.UI;

// One wire on the board (E3.2, Phase 3). It is a UI Graphic that draws a bezier ribbon between two
// ports (salvaged curve math from the Dev/Bagus prototype, rebuilt as a proper uGUI mesh). It holds
// the connection's identity so WiringController can remove it, and follows the ports every frame so
// a dragged node keeps its wires attached. A pending wire (no target yet) tracks the pointer.
//
// raycastTarget is off: the ribbon's rect fills the whole layer, so leaving it clickable would
// block every node. Cutting is done by right-clicking a port instead (see PortView / WiringController).
[RequireComponent(typeof(CanvasRenderer))]
public class ConnectionView : Graphic
{
    public PortView From { get; private set; }
    public PortView To { get; private set; }

    // Connection identity in the model (set once permanent); used for removal.
    public string FromNodeId, FromPortId, ToNodeId, ToPortId;

    public bool IsPending => To == null;

    private const float Thickness = 6f;
    private const int Segments = 20;

    private Vector2 freeScreenPoint; // where a pending wire's loose end sits
    private Vector2 localA, localB;

    protected override void Awake()
    {
        base.Awake();
        raycastTarget = false;
    }

    public void SetEndpoints(PortView from, PortView to)
    {
        From = from;
        To = to;
    }

    public void SetConnection(string fromNode, string fromPort, string toNode, string toPort)
    {
        FromNodeId = fromNode; FromPortId = fromPort; ToNodeId = toNode; ToPortId = toPort;
    }

    /// <summary>Move the loose end of a pending wire to a screen point (during a drag).</summary>
    public void SetFreeEnd(Vector2 screenPoint) => freeScreenPoint = screenPoint;

    void LateUpdate()
    {
        if (From == null) { Destroy(gameObject); return; }

        Vector2 aScreen = RectTransformUtility.WorldToScreenPoint(null, From.WorldPosition);
        Vector2 bScreen = To != null
            ? RectTransformUtility.WorldToScreenPoint(null, To.WorldPosition)
            : freeScreenPoint;

        RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform, aScreen, null, out localA);
        RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform, bScreen, null, out localB);
        SetVerticesDirty();
    }

    protected override void OnPopulateMesh(VertexHelper vh)
    {
        vh.Clear();
        if (From == null) return;

        float strength = Mathf.Max(40f, Mathf.Abs(localB.x - localA.x) * 0.5f);
        Vector2 c1 = localA + Vector2.right * strength;
        Vector2 c2 = localB + Vector2.left * strength;
        float half = Thickness * 0.5f;

        Vector2 prev = localA;
        for (int i = 1; i <= Segments; i++)
        {
            float t = i / (float)Segments;
            Vector2 cur = Bezier(t, localA, c1, c2, localB);
            Vector2 dir = cur - prev;
            if (dir.sqrMagnitude < 1e-6f) { prev = cur; continue; }
            Vector2 n = new Vector2(-dir.y, dir.x).normalized * half;

            int idx = vh.currentVertCount;
            vh.AddVert(prev - n, color, Vector2.zero);
            vh.AddVert(prev + n, color, Vector2.zero);
            vh.AddVert(cur + n, color, Vector2.zero);
            vh.AddVert(cur - n, color, Vector2.zero);
            vh.AddTriangle(idx, idx + 1, idx + 2);
            vh.AddTriangle(idx, idx + 2, idx + 3);
            prev = cur;
        }
    }

    private static Vector2 Bezier(float t, Vector2 p0, Vector2 p1, Vector2 p2, Vector2 p3)
    {
        float u = 1 - t;
        return u * u * u * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t * t * t * p3;
    }
}
