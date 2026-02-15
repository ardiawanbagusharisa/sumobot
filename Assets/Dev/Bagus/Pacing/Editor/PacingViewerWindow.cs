using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

public class PacingViewerWindow : EditorWindow
{
	private class Curve
	{
		public string name;
		public Color color;
		public List<float> values;
	}

	private List<Curve> curves = new List<Curve>();
	private const float padding = 30f;

	[MenuItem("Tools/Pacing Curve Viewer")]
	public static PacingViewerWindow Open() {
		return GetWindow<PacingViewerWindow>("Pacing Curve Viewer");
	}

	// 🔹 This is what you'll call
	public void SetCurves(
		List<float> list1,
		List<float> list2,
		List<float> list3) {
		curves.Clear();

		curves.Add(new Curve { name = "Curve Threats", color = Color.red, values = list1 });
		curves.Add(new Curve { name = "Curve Tempo", color = Color.cyan, values = list2 });
		curves.Add(new Curve { name = "Curve Overall", color = Color.green, values = list3 });

		Repaint();
	}

	private void OnGUI() {
		CurveTester tester = Selection.activeGameObject ?
			Selection.activeGameObject.GetComponent<CurveTester>() : null;

		if (tester == null) {
			EditorGUILayout.HelpBox("Select a GameObject with CurveTester.", MessageType.Info);
			return;
		}

		curves.Clear();

		curves.Add(new Curve { name = "Threats", color = Color.red, values = tester.Threats });
		curves.Add(new Curve { name = "Tempos", color = Color.cyan, values = tester.Tempos });
		curves.Add(new Curve { name = "Overall", color = Color.green, values = tester.Overall });

		Rect rect = GUILayoutUtility.GetRect(position.width, position.height - 20);
		EditorGUI.DrawRect(rect, new Color(0.15f, 0.15f, 0.15f));

		DrawCurves(rect);

		Repaint(); // keep updating
	}


	private void DrawCurves(Rect rect) {
		Handles.BeginGUI();

		float min = float.MaxValue;
		float max = float.MinValue;
		int maxCount = 0;

		foreach (var c in curves) {
			foreach (var v in c.values) {
				min = Mathf.Min(min, v);
				max = Mathf.Max(max, v);
			}

			maxCount = Mathf.Max(maxCount, c.values.Count);
		}

		if (Mathf.Approximately(min, max)) {
			min -= 1f;
			max += 1f;
		}

		float width = rect.width - padding * 2;
		float height = rect.height - padding * 2;

		foreach (var c in curves) {
			if (c.values.Count < 2)
				continue;

			Handles.color = c.color;

			for (int i = 1; i < c.values.Count; i++) {
				float x0 = rect.x + padding + width * ((i - 1f) / (maxCount - 1f));
				float x1 = rect.x + padding + width * (i / (maxCount - 1f));

				float y0 = Normalize(c.values[i - 1], min, max, rect, height);
				float y1 = Normalize(c.values[i], min, max, rect, height);

				Handles.DrawLine(new Vector3(x0, y0), new Vector3(x1, y1));
			}
		}

		Handles.EndGUI();
	}

	private float Normalize(float value, float min, float max, Rect rect, float height) {
		float t = Mathf.InverseLerp(min, max, value);
		return rect.y + rect.height - padding - t * height;
	}
}
