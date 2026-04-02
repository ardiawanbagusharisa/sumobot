using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class Connection : MonoBehaviour
{
	public enum LineStyle { Bezier, Orthogonal, Angled }
	public LineStyle style = LineStyle.Bezier;

	public PortUI from;
	public PortUI to;
	public float lineWidth = 2f;
	public Color lineColor = Color.white;
	public float curveStrength = 50f;
	public int segments = 15;
	public int capVertices = 5;

	[HideInInspector] public bool isInvalidPlacement = false;

	private LineRenderer lr;

	void Awake() {
		lr = GetComponent<LineRenderer>();

		// Unity's default LineRenderer materials often do not support vertex colors. 
		// assigning the built-in "Sprites/Default" material ensures startColor/endColor work.
		if (lr.sharedMaterial == null || lr.sharedMaterial.name.Contains("Default")) {
			lr.material = new Material(Shader.Find("Sprites/Default"));
		}

		lr.widthMultiplier = lineWidth;
		lr.numCapVertices = capVertices;
		lr.numCornerVertices = capVertices; // Smooths out the joints on angled/orthogonal lines
		lr.alignment = LineAlignment.TransformZ; // Prevents the 2D line from twisting by locking it to face the Z axis
		SetColor(lineColor);
	}

	public void SetColor(Color color) {
		lineColor = color;
		ApplyColor();
	}

	void ApplyColor() {
		Color color = isInvalidPlacement ? new Color(1, 0.2f, 0.2f, 0.5f) : lineColor;
		if (lr != null) {
			lr.startColor = color;
			lr.endColor = color;

			if (lr.material != null && lr.material.HasProperty("_Color")) {
				lr.material.color = color;
			}
		}
	}

	public void RemoveConnection() {
		if (from != null) from.connections.Remove(this);
		if (to != null) to.connections.Remove(this);
		Destroy(gameObject);
	}

	void Update() {
		if (Input.GetMouseButtonDown(1) && from != null && to != null) {
			Vector3[] points = new Vector3[lr.positionCount];
			lr.GetPositions(points);
			Vector3 mousePos = Input.mousePosition;

			Camera cam = null;
			Canvas canvas = GetComponentInParent<Canvas>();
			if (canvas != null && canvas.renderMode != RenderMode.ScreenSpaceOverlay) {
				cam = Camera.main;
			}

			for (int i = 0; i < points.Length - 1; i++) {
				Vector3 pA = cam == null ? points[i] : cam.WorldToScreenPoint(points[i]);
				Vector3 pB = cam == null ? points[i + 1] : cam.WorldToScreenPoint(points[i + 1]);

				float dist = DistancePointToSegment((Vector2)mousePos, (Vector2)pA, (Vector2)pB);

				if (dist < 15f) {
					RemoveConnection();
					break;
				}
			}
		}
	}

	float DistancePointToSegment(Vector2 pt, Vector2 p1, Vector2 p2) {
		float dx = p2.x - p1.x;
		float dy = p2.y - p1.y;
		if (dx == 0 && dy == 0) return Vector2.Distance(pt, p1);

		float t = ((pt.x - p1.x) * dx + (pt.y - p1.y) * dy) / (dx * dx + dy * dy);
		if (t < 0) return Vector2.Distance(pt, p1);
		if (t > 1) return Vector2.Distance(pt, p2);
		return Vector2.Distance(pt, new Vector2(p1.x + t * dx, p1.y + t * dy));
	}

	// Called manually to update the line position instead of using Update()
	public void RefreshLine() {
		if (from == null || to == null) return;
		DrawLine(from.GetWorldPosition(), to.GetWorldPosition());
		CheckLineCollision();
	}

	public void DrawToPoint(Vector3 endPoint) {
		if (from == null) return;
		DrawLine(from.GetWorldPosition(), endPoint);
		CheckLineCollision();
	}

	public void CheckLineCollision() {
		if (lr == null || lr.positionCount < 2) return;
		isInvalidPlacement = false;

		Vector3[] points = new Vector3[lr.positionCount];
		lr.GetPositions(points);

		foreach (var module in FindObjectsByType<ModuleUI>(FindObjectsSortMode.None)) {
			Rect rect = module.GetWorldRect();

			// Deflate rect by 5% to prevent line starts/ends directly on the node edges from triggering false collisions
			float shrinkX = rect.width * 0.05f;
			float shrinkY = rect.height * 0.05f;
			rect = new Rect(rect.x + shrinkX, rect.y + shrinkY, rect.width - shrinkX * 2, rect.height - shrinkY * 2);

			for (int i = 0; i < points.Length - 1; i++) {
				if (LineIntersectsRect(points[i], points[i + 1], rect)) {
					isInvalidPlacement = true;
					break;
				}
			}
			if (isInvalidPlacement) break;
		}

		ApplyColor();
	}

	bool LineIntersectsRect(Vector3 p1, Vector3 p2, Rect r) {
		if (r.Contains(p1) || r.Contains(p2)) return true;

		Vector3 r0 = new Vector3(r.xMin, r.yMin);
		Vector3 r1 = new Vector3(r.xMin, r.yMax);
		Vector3 r2 = new Vector3(r.xMax, r.yMax);
		Vector3 r3 = new Vector3(r.xMax, r.yMin);

		return LineIntersectsLine(p1, p2, r0, r1) ||
			   LineIntersectsLine(p1, p2, r1, r2) ||
			   LineIntersectsLine(p1, p2, r2, r3) ||
			   LineIntersectsLine(p1, p2, r3, r0);
	}

	bool LineIntersectsLine(Vector3 p1, Vector3 p2, Vector3 p3, Vector3 p4) {
		float numA = (p4.x - p3.x) * (p1.y - p3.y) - (p4.y - p3.y) * (p1.x - p3.x);
		float numB = (p2.x - p1.x) * (p1.y - p3.y) - (p2.y - p1.y) * (p1.x - p3.x);
		float den = (p4.y - p3.y) * (p2.x - p1.x) - (p4.x - p3.x) * (p2.y - p1.y);

		if (den == 0) return false;

		float uA = numA / den;
		float uB = numB / den;

		return (uA >= 0 && uA <= 1 && uB >= 0 && uB <= 1);
	}

	void DrawLine(Vector3 start, Vector3 end) {
		switch (style) {
			case LineStyle.Bezier:
				DrawBezier(start, end);
				break;
			case LineStyle.Orthogonal:
				DrawOrthogonal(start, end);
				break;
			case LineStyle.Angled:
				DrawAngled(start, end);
				break;
		}
	}

	void DrawOrthogonal(Vector3 p0, Vector3 p3) {
		float avoidDist = curveStrength;

		if (p0.x <= p3.x) {
			// Forward routing
			float midX = (p0.x + p3.x) / 2f;
			lr.positionCount = 4;
			lr.SetPositions(new Vector3[] { p0, new Vector3(midX, p0.y, p0.z), new Vector3(midX, p3.y, p3.z), p3 });
		} else {
			// Backward routing (wrap around)
			float midY = (p0.y + p3.y) / 2f;
			if (Mathf.Abs(p0.y - p3.y) < avoidDist) {
				midY = Mathf.Max(p0.y, p3.y) + avoidDist;
			}

			lr.positionCount = 6;
			lr.SetPositions(new Vector3[] {
				p0,
				new Vector3(p0.x + avoidDist, p0.y, p0.z),
				new Vector3(p0.x + avoidDist, midY, p0.z),
				new Vector3(p3.x - avoidDist, midY, p3.z),
				new Vector3(p3.x - avoidDist, p3.y, p3.z),
				p3
			});
		}
	}

	void DrawAngled(Vector3 p0, Vector3 p3) {
		float avoidDist = curveStrength;

		if (p0.x <= p3.x) {
			// Forward routing
			float dx = p3.x - p0.x;
			float dy = Mathf.Abs(p3.y - p0.y);

			if (dx >= dy) {
				// Room for true 45-degree angle
				float halfX = (dx - dy) / 2f;
				Vector3 p1 = new Vector3(p0.x + halfX, p0.y, p0.z);
				Vector3 p2 = new Vector3(p3.x - halfX, p3.y, p3.z);
				lr.positionCount = 4;
				lr.SetPositions(new Vector3[] { p0, p1, p2, p3 });
			} else {
				// Too close horizontally for 45 deg, use range of angles (steeper slope)
				float stub = Mathf.Min(dx / 3f, avoidDist / 2f);
				Vector3 p1 = new Vector3(p0.x + stub, p0.y, p0.z);
				Vector3 p2 = new Vector3(p3.x - stub, p3.y, p3.z);
				lr.positionCount = 4;
				lr.SetPositions(new Vector3[] { p0, p1, p2, p3 });
			}
		} else {
			// Backward routing (angled wrap around)
			float midY = (p0.y + p3.y) / 2f;
			if (Mathf.Abs(p0.y - p3.y) < avoidDist) {
				midY = Mathf.Max(p0.y, p3.y) + avoidDist;
			}

			float dyHalf = Mathf.Abs(midY - p0.y);

			Vector3 p1 = new Vector3(p0.x + avoidDist, p0.y, p0.z);
			Vector3 p4 = new Vector3(p3.x - avoidDist, p3.y, p3.z);

			float slantX = Mathf.Min(avoidDist, dyHalf);

			Vector3 p2 = new Vector3(p1.x - slantX, midY, p0.z);
			Vector3 p3_p = new Vector3(p4.x + slantX, midY, p3.z);

			lr.positionCount = 6;
			lr.SetPositions(new Vector3[] { p0, p1, p2, p3_p, p4, p3 });
		}
	}

	// Make sure it draws once it's completely hooked up
	void Start() {
		RefreshLine();
	}

	void DrawBezier(Vector3 p0, Vector3 p3) {
		Vector3 p1 = p0 + Vector3.right * curveStrength;
		Vector3 p2 = p3 + Vector3.left * curveStrength;

		lr.positionCount = segments;
		for (int i = 0; i < segments; i++) {
			float t = i / (float)(segments - 1);
			Vector3 point = Bezier(t, p0, p1, p2, p3);
			lr.SetPosition(i, point);
		}
	}

	Vector3 Bezier(float t, Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3) {
		return Mathf.Pow(1 - t, 3) * p0 +
			   3 * Mathf.Pow(1 - t, 2) * t * p1 +
			   3 * (1 - t) * t * t * p2 +
			   t * t * t * p3;
	}
}