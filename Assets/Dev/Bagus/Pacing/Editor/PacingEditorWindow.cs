using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

public class PacingEditorWindow : EditorWindow
{
	private const int DATA_COUNT = 25;
	private const float padding = 50f;
	private const float pointSize = 8f;

	private List<float> threats = new List<float>();
	private List<float> tempos = new List<float>();
	private List<float> overall = new List<float>();

	private int draggingCurve = -1;
	private int draggingIndex = -1;

	[MenuItem("Tools/Pacing Curve Editor")]
	public static void Open() {
		GetWindow<PacingEditorWindow>("Pacing Curve Editor");
	}

	private void OnEnable() {
		GenerateSampleData();
	}

	private void GenerateSampleData() {
		threats.Clear();
		tempos.Clear();
		overall.Clear();

		for (int i = 0; i < DATA_COUNT; i++) {
			float t = i / (float)(DATA_COUNT - 1);

			float threat = Mathf.Clamp01(Mathf.Sin(t * Mathf.PI) * 0.5f + 0.5f);
			float tempo = Mathf.Clamp01(Mathf.PerlinNoise(i * 0.2f, 0f));
			float over = (threat + tempo) * 0.5f;

			threats.Add(threat);
			tempos.Add(tempo);
			overall.Add(over);
		}
	}

	private void OnGUI() {
		Rect rect = GUILayoutUtility.GetRect(position.width, position.height - 20);

		EditorGUI.DrawRect(rect, new Color(0.12f, 0.12f, 0.12f));

		DrawGrid(rect);   // NEW: draw grid
		DrawAxes(rect);   // axes lines & labels
		DrawAndEdit(rect); // curves + draggable circles

		Repaint();
	}

	private void DrawGrid(Rect rect) {
		Handles.BeginGUI();
		Color gridColor = new Color(0.3f, 0.3f, 0.3f, 0.5f);
		Handles.color = gridColor;

		float left = rect.x + padding;
		float right = rect.x + rect.width - padding;
		float top = rect.y + padding;
		float bottom = rect.y + rect.height - padding;

		float width = right - left;
		float height = bottom - top;

		// ---------- Horizontal grid lines ----------
		int ySteps = 4; // 0, 0.25, 0.5, 0.75, 1
		for (int i = 0; i <= ySteps; i++) {
			float t = i / (float)ySteps;
			float y = bottom - t * height;

			Handles.DrawLine(new Vector3(left, y), new Vector3(right, y));

			// Y Labels
			GUI.Label(new Rect(left - 40, y - 10, 35, 20), t.ToString("0.##"));
		}

		// ---------- Vertical grid lines ----------
		int xSteps = DATA_COUNT - 1;
		for (int i = 0; i <= xSteps; i++) {
			float x = left + i / (float)xSteps * width;
			Handles.DrawLine(new Vector3(x, top), new Vector3(x, bottom));

			// X Labels (every 5 points to avoid clutter)
			if (i % 5 == 0 || i == 0 || i == xSteps)
				GUI.Label(new Rect(x - 5, bottom + 2, 30, 20), i.ToString());
		}

		Handles.EndGUI();
	}


	private void DrawAxes(Rect rect) {
		Handles.BeginGUI();
		Handles.color = Color.gray;

		float left = rect.x + padding;
		float right = rect.x + rect.width - padding;
		float top = rect.y + padding;
		float bottom = rect.y + rect.height - padding;

		// Y Axis
		Handles.DrawLine(new Vector3(left, top), new Vector3(left, bottom));

		// X Axis
		Handles.DrawLine(new Vector3(left, bottom), new Vector3(right, bottom));

		Handles.EndGUI();

		// Labels - Should only turn this on when the grid is off. 
		//GUI.Label(new Rect(left - 30, top - 10, 40, 20), "1");
		//GUI.Label(new Rect(left - 30, bottom - 10, 40, 20), "0");

		//GUI.Label(new Rect(left - 10, bottom + 5, 40, 20), "0");
		//GUI.Label(new Rect(right - 20, bottom + 5, 40, 20), (DATA_COUNT - 1).ToString());
	}

	private void DrawAndEdit(Rect rect) {
		Handles.BeginGUI();

		float left = rect.x + padding;
		float right = rect.x + rect.width - padding;
		float top = rect.y + padding;
		float bottom = rect.y + rect.height - padding;

		float width = right - left;
		float height = bottom - top;

		List<List<float>> lists = new List<List<float>>
		{
		threats,
		tempos,
		overall
	};

		Color[] colors = { Color.red, Color.cyan, Color.green };

		Event e = Event.current;

		for (int c = 0; c < lists.Count; c++) {
			var list = lists[c];
			Handles.color = colors[c];

			// ---------- Compute all screen points ----------
			Vector2[] screenPoints = new Vector2[DATA_COUNT];

			for (int i = 0; i < DATA_COUNT; i++) {
				float normalizedX = i / (float)(DATA_COUNT - 1);
				float x = left + normalizedX * width;

				float normalizedY = Mathf.Clamp01(list[i]);
				float y = bottom - normalizedY * height;

				screenPoints[i] = new Vector2(x, y);
			}

			// ---------- Draw lines ----------
			for (int i = 0; i < DATA_COUNT - 1; i++) {
				Handles.DrawLine(screenPoints[i], screenPoints[i + 1]);
			}

			// ---------- Draw draggable points as circles ----------
			float radius = pointSize / 2f; // radius in pixels

			for (int i = 0; i < DATA_COUNT; i++) {
				Vector2 p = screenPoints[i];

				// Draw circle
				Handles.color = colors[c];
				Handles.DrawSolidDisc(p, Vector3.forward, radius);

				// Handle dragging
				if (e.type == EventType.MouseDown && Vector2.Distance(e.mousePosition, p) < radius) {
					draggingCurve = c;
					draggingIndex = i;
					e.Use();
				}

				if (e.type == EventType.MouseDrag &&
					draggingCurve == c &&
					draggingIndex == i) {
					float normalized = Mathf.Clamp01((bottom - e.mousePosition.y) / height);
					list[i] = normalized;
					e.Use();
				}
			}
		}

		if (e.type == EventType.MouseUp) {
			draggingCurve = -1;
			draggingIndex = -1;
		}

		Handles.EndGUI();
	}

	//private void DrawAndEdit(Rect rect) {
	//	Handles.BeginGUI();

	//	float left = rect.x + padding;
	//	float right = rect.x + rect.width - padding;
	//	float top = rect.y + padding;
	//	float bottom = rect.y + rect.height - padding;

	//	float width = right - left;
	//	float height = bottom - top;

	//	List<List<float>> lists = new List<List<float>>
	//	{
	//	threats,
	//	tempos,
	//	overall
	//};

	//	Color[] colors = { Color.red, Color.cyan, Color.green };

	//	Event e = Event.current;

	//	for (int c = 0; c < lists.Count; c++) {
	//		var list = lists[c];
	//		Handles.color = colors[c];

	//		// ---------- 1️⃣ Compute all screen points first ----------
	//		Vector2[] screenPoints = new Vector2[DATA_COUNT];

	//		for (int i = 0; i < DATA_COUNT; i++) {
	//			float normalizedX = i / (float)(DATA_COUNT - 1);
	//			float x = left + normalizedX * width;

	//			float normalizedY = Mathf.Clamp01(list[i]);
	//			float y = bottom - normalizedY * height;

	//			screenPoints[i] = new Vector2(x, y);
	//		}

	//		// ---------- 2️⃣ Draw lines between consecutive points ----------
	//		for (int i = 0; i < DATA_COUNT - 1; i++) {
	//			Handles.DrawLine(screenPoints[i], screenPoints[i + 1]);
	//		}

	//		// ---------- 3️⃣ Draw draggable points ----------
	//		for (int i = 0; i < DATA_COUNT; i++) {
	//			Vector2 p = screenPoints[i];

	//			Rect pointRect = new Rect(
	//				p.x - pointSize / 2,
	//				p.y - pointSize / 2,
	//				pointSize,
	//				pointSize);

	//			EditorGUI.DrawRect(pointRect, colors[c]);

	//			if (e.type == EventType.MouseDown && pointRect.Contains(e.mousePosition)) {
	//				draggingCurve = c;
	//				draggingIndex = i;
	//				e.Use();
	//			}

	//			if (e.type == EventType.MouseDrag &&
	//				draggingCurve == c &&
	//				draggingIndex == i) {
	//				float normalized =
	//					Mathf.Clamp01((bottom - e.mousePosition.y) / height);

	//				list[i] = normalized;
	//				e.Use();
	//			}
	//		}
	//	}

	//	if (e.type == EventType.MouseUp) {
	//		draggingCurve = -1;
	//		draggingIndex = -1;
	//	}

	//	Handles.EndGUI();
	//}

}
