using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class Connection : MonoBehaviour
{
	public PortUI from;
	public PortUI to;
	public float lineWidth = 2f;
	public Color lineColor = Color.white;
	public float curveStrength = 50f;
	public int segments = 15;
	public int capVertices = 5;

	private LineRenderer lr;

	void Awake() {
		lr = GetComponent<LineRenderer>();
		lr.widthMultiplier = lineWidth;
		lr.numCapVertices = capVertices;
		SetColor(lineColor);
	}

	public void SetColor(Color color) {
		lineColor = color;
		if (lr != null) {
			lr.startColor = color;
			lr.endColor = color;
		}
	}

	// Called manually to update the line position instead of using Update()
	public void RefreshLine() {
		if (from == null || to == null) return;
		DrawBezier(from.GetWorldPosition(), to.GetWorldPosition());
	}

	public void DrawToPoint(Vector3 endPoint) {
		if (from == null) return;
		DrawBezier(from.GetWorldPosition(), endPoint);
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