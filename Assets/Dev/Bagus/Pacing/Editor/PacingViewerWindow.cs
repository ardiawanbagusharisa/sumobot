using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace PacingFramework
{
	public class PacingViewerWindow : EditorWindow
	{
		private PacingController controller;
		private PacingTargetConfig targetConfig;
		private string loadedConfigPath = "";

		private Vector2 scroll;
		private const float padding = 50f;
		private const float pointRadius = 4f;

		private bool overlayTarget = true;

		[MenuItem("Tools/Pacing Framework/Pacing Viewer")]
		public static void Open() {
			GetWindow<PacingViewerWindow>("Pacing Viewer");
		}

		private void OnGUI() {
			scroll = EditorGUILayout.BeginScrollView(scroll);

			DrawSelectionSection();

			if (controller == null) {
				EditorGUILayout.HelpBox("Assign a PacingController.", MessageType.Info);
				EditorGUILayout.EndScrollView();
				return;
			}

			GamePacing history = controller.GetHistory();

			if (history.SegmentPacings.Count == 0) {
				EditorGUILayout.HelpBox("No pacing data yet.", MessageType.Info);
				EditorGUILayout.EndScrollView();
				return;
			}

			List<float> threat = ExtractThreat(history);
			List<float> tempo = ExtractTempo(history);
			List<float> overall = ExtractOverall(history);

			Rect rect = GUILayoutUtility.GetRect(position.width - 20, 400);
			EditorGUI.DrawRect(rect, new Color(0.12f, 0.12f, 0.12f));

			DrawGrid(rect, threat.Count);
			DrawAxes(rect);

			DrawCurve(rect, threat, Color.red);
			DrawCurve(rect, tempo, Color.cyan);
			DrawCurve(rect, overall, Color.green);

			if (overlayTarget && targetConfig != null) {
				DrawTargetOverlay(rect, targetConfig);
				DrawEvaluation(threat, tempo);
			}

			DrawLegend(rect);

			Repaint(); // live update

			EditorGUILayout.EndScrollView();
		}

		// ======================================================
		// UI
		// ======================================================

		private void DrawSelectionSection() {
			EditorGUILayout.BeginVertical("box");

			controller = (PacingController)EditorGUILayout.ObjectField(
				"Pacing Controller",
				controller,
				typeof(PacingController),
				true);

			EditorGUILayout.BeginVertical("box");

			EditorGUILayout.BeginHorizontal();

			EditorGUILayout.LabelField("Target Config (Optional)", GUILayout.Width(160));

			EditorGUI.BeginDisabledGroup(true);
			EditorGUILayout.TextField("Loaded Path", loadedConfigPath);
			EditorGUI.EndDisabledGroup();

			if (GUILayout.Button("Load JSON", GUILayout.Width(100))) {
				string path = EditorUtility.OpenFilePanel(
					"Load Target Config",
					"",
					"json");

				if (!string.IsNullOrEmpty(path)) {
					string json = System.IO.File.ReadAllText(path);
					targetConfig = JsonUtility.FromJson<PacingTargetConfig>(json);
					loadedConfigPath = path;
				}
			}

			EditorGUILayout.EndHorizontal();
			EditorGUILayout.EndVertical();

			overlayTarget = EditorGUILayout.Toggle("Overlay Target", overlayTarget);

			EditorGUILayout.EndVertical();
		}

		// ======================================================
		// DATA EXTRACTION
		// ======================================================

		private List<float> ExtractThreat(GamePacing history) {
			var list = new List<float>();
			foreach (var p in history.SegmentPacings)
				list.Add(p.Threat.Value);
			return list;
		}

		private List<float> ExtractTempo(GamePacing history) {
			var list = new List<float>();
			foreach (var p in history.SegmentPacings)
				list.Add(p.Tempo.Value);
			return list;
		}

		private List<float> ExtractOverall(GamePacing history) {
			var list = new List<float>();
			foreach (var p in history.SegmentPacings)
				list.Add(p.GetOverallPacing());
			return list;
		}

		// ======================================================
		// DRAWING
		// ======================================================

		private void DrawGrid(Rect rect, int count) {
			Handles.BeginGUI();
			Handles.color = new Color(0.3f, 0.3f, 0.3f, 0.4f);

			float left = rect.x + padding;
			float right = rect.x + rect.width - padding;
			float top = rect.y + padding;
			float bottom = rect.y + rect.height - padding;

			float width = right - left;
			float height = bottom - top;

			int xSteps = Mathf.Max(1, count - 1);

			for (int i = 0; i <= xSteps; i++) {
				float x = left + i / (float)xSteps * width;
				Handles.DrawLine(new Vector3(x, top), new Vector3(x, bottom));

				if (i % 5 == 0 || i == 0 || i == xSteps)
					GUI.Label(new Rect(x - 10, bottom + 2, 40, 20), i.ToString());
			}

			for (int i = 0; i <= 4; i++) {
				float t = i / 4f;
				float y = bottom - t * height;
				Handles.DrawLine(new Vector3(left, y), new Vector3(right, y));

				GUI.Label(new Rect(left - 35, y - 8, 30, 20), t.ToString("0.##"));
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

			Handles.DrawLine(new Vector3(left, top), new Vector3(left, bottom));
			Handles.DrawLine(new Vector3(left, bottom), new Vector3(right, bottom));

			Handles.EndGUI();
		}

		private void DrawCurve(Rect rect, List<float> list, Color color) {
			if (list.Count < 2)
				return;

			Handles.BeginGUI();
			Handles.color = color;

			float left = rect.x + padding;
			float right = rect.x + rect.width - padding;
			float top = rect.y + padding;
			float bottom = rect.y + rect.height - padding;

			float width = right - left;
			float height = bottom - top;

			Vector2 GetPoint(int i) {
				float x = left + i / (float)(list.Count - 1) * width;
				float y = bottom - Mathf.Clamp01(list[i]) * height;
				return new Vector2(x, y);
			}

			for (int i = 0; i < list.Count - 1; i++)
				Handles.DrawLine(GetPoint(i), GetPoint(i + 1));

			for (int i = 0; i < list.Count; i++)
				Handles.DrawSolidDisc(GetPoint(i), Vector3.forward, pointRadius);

			Handles.EndGUI();
		}

		private void DrawTargetOverlay(Rect rect, PacingTargetConfig target) {
			DrawDashed(rect, target.ThreatTargets, Color.red);
			DrawDashed(rect, target.TempoTargets, Color.cyan);
		}

		private void DrawDashed(Rect rect, List<float> list, Color color) {
			if (list == null || list.Count < 2)
				return;

			Handles.BeginGUI();
			Handles.color = color;

			float left = rect.x + padding;
			float right = rect.x + rect.width - padding;
			float top = rect.y + padding;
			float bottom = rect.y + rect.height - padding;

			float width = right - left;
			float height = bottom - top;

			Vector2 GetPoint(int i) {
				float x = left + i / (float)(list.Count - 1) * width;
				float y = bottom - Mathf.Clamp01(list[i]) * height;
				return new Vector2(x, y);
			}

			for (int i = 0; i < list.Count - 1; i++)
				Handles.DrawDottedLine(GetPoint(i), GetPoint(i + 1), 4f);

			Handles.EndGUI();
		}

		private void DrawLegend(Rect rect) {
			float boxWidth = 110f;
			float boxHeight = 70f;
			float margin = 10f;

			Rect legendRect = new Rect(
				rect.xMax - boxWidth - margin,
				rect.y + margin,
				boxWidth,
				boxHeight
			);

			EditorGUI.DrawRect(legendRect, new Color(0f, 0f, 0f, 0.4f));

			DrawLegendItem(legendRect, 0, Color.red, "Threat");
			DrawLegendItem(legendRect, 1, Color.cyan, "Tempo");
			DrawLegendItem(legendRect, 2, Color.green, "Overall");
			//DrawLegendItem(legendRect, 3, Color.white, "Dashed = Target");
		}

		private void DrawLegendItem(Rect legendRect, int row, Color color, string label) {
			float rowHeight = 18f;
			float y = legendRect.y + 6 + row * rowHeight;

			Rect colorRect = new Rect(legendRect.x + 6, y + 4, 12, 12);
			EditorGUI.DrawRect(colorRect, color);

			EditorGUI.LabelField(
				new Rect(legendRect.x + 24, y, 90, 20),
				label,
				EditorStyles.whiteLabel
			);
		}

		// ======================================================
		// EVALUATION
		// ======================================================

		private void DrawEvaluation(List<float> threat, List<float> tempo) {
			if (targetConfig == null) return;

			float threatError = CalculateMSE(threat, targetConfig.ThreatTargets);
			float tempoError = CalculateMSE(tempo, targetConfig.TempoTargets);

			EditorGUILayout.Space();
			EditorGUILayout.LabelField(
				$"Threat MSE: {threatError:F4}    Tempo MSE: {tempoError:F4}",
				EditorStyles.boldLabel);
		}

		private float CalculateMSE(List<float> a, List<float> b) {
			if (a == null || b == null) return 0f;

			int count = Mathf.Min(a.Count, b.Count);
			if (count == 0) return 0f;

			float error = 0f;
			for (int i = 0; i < count; i++) {
				float d = a[i] - b[i];
				error += d * d;
			}

			return error / count;
		}
	}
}