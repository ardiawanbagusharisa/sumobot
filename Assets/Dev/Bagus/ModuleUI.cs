using UnityEngine;
using UnityEngine.EventSystems;

[RequireComponent(typeof(RectTransform))]
public class ModuleUI : MonoBehaviour, IBeginDragHandler, IDragHandler, IEndDragHandler
{
	public RectTransform rect;
	public bool isInvalidPlacement = false;

	private Color originalColor;
	private readonly Color invalidColor = new Color(1,0.2f,0.2f,0.5f);
	private Canvas canvas;
	private Vector2 offset;

	void Awake() {
		rect = GetComponent<RectTransform>();
		canvas = GetComponentInParent<Canvas>();
		originalColor = GetComponent<UnityEngine.UI.Image>().color;
	}

	public void OnBeginDrag(PointerEventData eventData) {
		RectTransformUtility.ScreenPointToLocalPointInRectangle(
			rect, eventData.position, eventData.pressEventCamera, out offset);
	}

	public void OnDrag(PointerEventData eventData) {
		Vector2 localPoint;
		RectTransformUtility.ScreenPointToLocalPointInRectangle(
			canvas.transform as RectTransform,
			eventData.position,
			eventData.pressEventCamera,
			out localPoint);

		rect.anchoredPosition = localPoint - offset;

		// Update colors for all modules dynamically during drag
		foreach (var module in FindObjectsByType<ModuleUI>(FindObjectsSortMode.None)) {
			module.CheckCollision();
		}

		UpdatePortConnections();

		foreach (var conn in FindObjectsByType<Connection>(FindObjectsSortMode.None)) {
			conn.CheckLineCollision();
		}
	}

	public void OnEndDrag(PointerEventData eventData) {
		// Re-evaluate collisions for all modules to update their states and colors
		foreach (var module in FindObjectsByType<ModuleUI>(FindObjectsSortMode.None)) {
			module.CheckCollision();
		}
		UpdatePortConnections();

		foreach (var conn in FindObjectsByType<Connection>(FindObjectsSortMode.None)) {
			conn.CheckLineCollision();
		}
	}

	void UpdatePortConnections() {
		// Update all connections for ports on this module
		PortUI[] ports = GetComponentsInChildren<PortUI>();
		foreach (PortUI port in ports) {
			foreach (var conn in port.connections) {
				if (conn != null) {
					conn.RefreshLine();
				}
			}
		}
	}

	void CheckCollision() {
		isInvalidPlacement = false;

		foreach (var other in FindObjectsByType<ModuleUI>(FindObjectsSortMode.None)) {
			if (other == this) continue;

			if (RectOverlaps(rect, other.rect)) {
				isInvalidPlacement = true;
				break;
			}
		}

		GetComponent<UnityEngine.UI.Image>().color =
			isInvalidPlacement ? invalidColor : originalColor;
	}

	bool RectOverlaps(RectTransform a, RectTransform b) {
		Rect rectA = GetWorldRect(a);
		Rect rectB = GetWorldRect(b);
		return rectA.Overlaps(rectB);
	}

	public Rect GetWorldRect() {
		return GetWorldRect(rect);
	}

	Rect GetWorldRect(RectTransform rt) {
		Vector3[] corners = new Vector3[4];
		rt.GetWorldCorners(corners);

		return new Rect(
			corners[0].x,
			corners[0].y,
			corners[2].x - corners[0].x,
			corners[2].y - corners[0].y
		);
	}
}