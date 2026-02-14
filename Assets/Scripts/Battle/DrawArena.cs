using UnityEngine;

[ExecuteAlways]
public class DrawArena : MonoBehaviour
{
    public enum TargetType { 
        Auto, Circle, Polygon, Box 
    }

    public Transform arenaCollider;
    public SpriteRenderer fill;
    public LineRenderer outline; 

    public TargetType targetShape = TargetType.Auto;
    
    public Color lineColor = Color.white;
    [Min(0.0f)] public float lineWidth = 0.1f; // world units
    [Range(8, 360)] public int circleSegments = 180;
    [Range(0, 8)] public int polygonEdgeSubdivisions = 0; 
    public string sortingLayerName = "Default";
    public int sortingOrder = 5;

    void Reset()
    {
        outline.useWorldSpace = true;
        outline.loop = true;
        outline.widthMultiplier = Mathf.Max(0.0f, lineWidth);
        outline.alignment = LineAlignment.View; // face camera in 2D
        outline.numCapVertices = 2;
        outline.numCornerVertices = 2;

        if (!outline.sharedMaterial)
            outline.sharedMaterial = new Material(Shader.Find("Sprites/Default"));

        outline.startColor = outline.endColor = lineColor;
        outline.sortingLayerName = sortingLayerName;
        outline.sortingOrder = sortingOrder;
    }

    void OnEnable()
    {
        EnsureLR();
        Rebuild();
    }

    void OnValidate()
    {
        EnsureLR();
        ApplyLRStyle();
        Rebuild();
    }

    void LateUpdate()
    {
        Rebuild();
    }

    void EnsureLR()
    {
        if (!outline) outline = GetComponent<LineRenderer>();
        outline.enabled = true;
        if (!outline.sharedMaterial)
            outline.sharedMaterial = new Material(Shader.Find("Sprites/Default"));
    }

    void ApplyLRStyle()
    {
        outline.widthMultiplier = Mathf.Max(0.0f, lineWidth);
        outline.startColor = outline.endColor = lineColor;
        outline.sortingLayerName = sortingLayerName;
        outline.sortingOrder = sortingOrder;
        outline.alignment = LineAlignment.View;
        outline.loop = true;
        outline.useWorldSpace = true;
    }

    void Rebuild()
    {
        var col = PickCollider();
        if (!col) { outline.positionCount = 0; return; }

        var t = col.transform;

        if ((targetShape == TargetType.Auto || targetShape == TargetType.Polygon) && col is PolygonCollider2D pc)
        {
            DrawPolygon(pc, t);
        }
        else if ((targetShape == TargetType.Auto || targetShape == TargetType.Box) && col is BoxCollider2D bc)
        {
            DrawBox(bc, t);
        }
        else if ((targetShape == TargetType.Auto || targetShape == TargetType.Circle) && col is CircleCollider2D cc)
        {
            DrawCircle(cc, t);
        }
        else
        {
            outline.positionCount = 0;
        }
    }

    Collider2D PickCollider()
    {
        Transform root = arenaCollider ? arenaCollider : (transform.parent ? transform.parent : transform);
        if (!root) return null;

        var all = root.GetComponents<Collider2D>();
        if (all == null || all.Length == 0) return null;

        if (targetShape == TargetType.Circle)
            foreach (var c in all) if (c is CircleCollider2D && c.enabled) return c;

        if (targetShape == TargetType.Box)
            foreach (var c in all) if (c is BoxCollider2D && c.enabled) return c;

        if (targetShape == TargetType.Polygon)
            foreach (var c in all) if (c is PolygonCollider2D && c.enabled) return c;

        // Auto priority: Polygon > Box > Circle
        foreach (var c in all) if (c is PolygonCollider2D && c.enabled) return c;
        foreach (var c in all) if (c is BoxCollider2D && c.enabled) return c;
        foreach (var c in all) if (c is CircleCollider2D && c.enabled) return c;

        // fallback to any enabled collider
        foreach (var c in all) if (c.enabled) return c;
        return null;
    }

    // ---- Circle / Ellipse (handles non-uniform scale) ----
    void DrawCircle(CircleCollider2D cc, Transform tr)
    {
        int n = Mathf.Clamp(circleSegments, 8, 2048);
        outline.positionCount = n;

        Vector2 localCenter = cc.offset;
        Vector3 worldCenter = tr.TransformPoint((Vector3)localCenter);

        float rx = cc.radius * Mathf.Abs(tr.lossyScale.x);
        float ry = cc.radius * Mathf.Abs(tr.lossyScale.y);

        Vector3 ax = tr.right; // local X in world
        Vector3 ay = tr.up;    // local Y in world

        float step = 2f * Mathf.PI / n;
        for (int i = 0; i < n; i++)
        {
            float a = i * step;
            Vector3 p = worldCenter + ax * (rx * Mathf.Cos(a)) + ay * (ry * Mathf.Sin(a));
            outline.SetPosition(i, p);
        }
    }

    // ---- Polygon (outer path by default) ----
    void DrawPolygon(PolygonCollider2D pc, Transform tr)
    {
        if (pc.pathCount == 0) { outline.positionCount = 0; return; }

        var pts = pc.GetPath(0); // outermost
        if (pts == null || pts.Length < 2) { outline.positionCount = 0; return; }

        int segs = Mathf.Max(0, polygonEdgeSubdivisions);
        int total = pts.Length * (segs + 1);
        outline.positionCount = total;

        int k = 0;
        for (int i = 0; i < pts.Length; i++)
        {
            Vector2 a = pts[i] + pc.offset;
            Vector2 b = pts[(i + 1) % pts.Length] + pc.offset;

            if (segs == 0)
            {
                outline.SetPosition(k++, tr.TransformPoint(a));
            }
            else
            {
                for (int s = 0; s <= segs; s++)
                {
                    float t = (float)s / (segs + 1); // < 1, next edge supplies its vertex
                    Vector2 p = Vector2.Lerp(a, b, t);
                    outline.SetPosition(k++, tr.TransformPoint(p));
                }
            }
        }
    }

    // ---- Box (local axis-aligned; transform handles rotation/scale) ----
    void DrawBox(BoxCollider2D bc, Transform tr)
    {
        Vector2 half = bc.size * 0.5f;
        Vector2 off = bc.offset;

        Vector3[] local = new Vector3[4] {
            new Vector3(off.x - half.x, off.y - half.y, 0), // BL
            new Vector3(off.x + half.x, off.y - half.y, 0), // BR
            new Vector3(off.x + half.x, off.y + half.y, 0), // TR
            new Vector3(off.x - half.x, off.y + half.y, 0)  // TL
        };

        int segs = Mathf.Max(0, polygonEdgeSubdivisions);
        int verts = 4 * (segs + 1);
        outline.positionCount = verts;

        int k = 0;
        for (int i = 0; i < 4; i++)
        {
            Vector3 a = local[i];
            Vector3 b = local[(i + 1) % 4];

            if (segs == 0)
            {
                outline.SetPosition(k++, tr.TransformPoint(a));
            }
            else
            {
                for (int s = 0; s <= segs; s++)
                {
                    float t = (float)s / (segs + 1); // < 1
                    Vector3 p = Vector3.Lerp(a, b, t);
                    outline.SetPosition(k++, tr.TransformPoint(p));
                }
            }
        }
    }

    // ---- Gizmo debug (draws same as LR) ----
    void OnDrawGizmosSelected()
    {
        var col = PickCollider();
        if (!col) return;

        Gizmos.color = new Color(lineColor.r, lineColor.g, lineColor.b, 0.5f);

        if (col is CircleCollider2D cc)
        {
            int n = Mathf.Clamp(circleSegments, 16, 256);
            Vector2 c = cc.offset;
            Vector3 wc = col.transform.TransformPoint((Vector3)c);
            float rx = cc.radius * Mathf.Abs(col.transform.lossyScale.x);
            float ry = cc.radius * Mathf.Abs(col.transform.lossyScale.y);
            Vector3 ax = col.transform.right, ay = col.transform.up;
            float step = 2f * Mathf.PI / n;
            Vector3 prev = wc + ax * rx;
            for (int i = 1; i <= n; i++)
            {
                float a = i * step;
                Vector3 p = wc + ax * (rx * Mathf.Cos(a)) + ay * (ry * Mathf.Sin(a));
                Gizmos.DrawLine(prev, p);
                prev = p;
            }
        }
        else if (col is PolygonCollider2D pc)
        {
            var pts = pc.GetPath(0);
            if (pts.Length < 2) return;
            Vector3 prev = col.transform.TransformPoint((Vector3)(pts[0] + pc.offset));
            for (int i = 1; i <= pts.Length; i++)
            {
                Vector3 curr = col.transform.TransformPoint((Vector3)(pts[i % pts.Length] + pc.offset));
                Gizmos.DrawLine(prev, curr);
                prev = curr;
            }
        }
        else if (col is BoxCollider2D bc)
        {
            Vector2 h = bc.size * 0.5f; Vector2 off = bc.offset;
            Vector3[] p = new Vector3[5];
            p[0] = col.transform.TransformPoint(new Vector3(off.x - h.x, off.y - h.y));
            p[1] = col.transform.TransformPoint(new Vector3(off.x + h.x, off.y - h.y));
            p[2] = col.transform.TransformPoint(new Vector3(off.x + h.x, off.y + h.y));
            p[3] = col.transform.TransformPoint(new Vector3(off.x - h.x, off.y + h.y));
            p[4] = p[0];
            for (int i = 0; i < 4; i++) Gizmos.DrawLine(p[i], p[i + 1]);
        }
    }
}
