using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace PacingFramework
{
	public class PacingEditorWindow : EditorWindow
	{
		private bool showConstraints = true;
		private bool showSegmentFields = true;

		private ConstraintConfig constraintConfig = new ConstraintConfig();
		private Vector2 scrollPos;
		private string saveFileName = "PacingConfig.json";

		private const int DATA_COUNT = 25;
		private const float padding = 50f;
		private const float pointSize = 10f;

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
			scrollPos = EditorGUILayout.BeginScrollView(scrollPos);

			DrawGraphSection();
			EditorGUILayout.Space(20);

			DrawConstraintSection();
			EditorGUILayout.Space(20);

			DrawSaveSection();
			EditorGUILayout.Space(20);
			
			DrawLoadSection();


			EditorGUILayout.EndScrollView();

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

		private void DrawConstraintSection() {
			showConstraints = EditorGUILayout.Foldout(showConstraints, "Global Constraints", true);

			if (!showConstraints)
				return;

			EditorGUILayout.BeginVertical("box");

			DrawConstraintRow("Collision Ratio", constraintConfig.CollisionRatio);
			DrawConstraintRow("Ability Ratio", constraintConfig.AbilityRatio);
			DrawConstraintRow("Angle", constraintConfig.Angle);
			DrawConstraintRow("Safe Distance", constraintConfig.SafeDistance);

			EditorGUILayout.Space(8);

			DrawConstraintRow("Action Intensity", constraintConfig.ActionIntensity);
			DrawConstraintRow("Action Density", constraintConfig.ActionDensity);
			DrawConstraintRow("Bots Distance", constraintConfig.BotsDistance);
			DrawConstraintRow("Velocity", constraintConfig.Velocity);

			EditorGUILayout.EndVertical();
		}

		private void DrawConstraintRow(string label, FloatMinMax data) {
			EditorGUILayout.BeginHorizontal();

			GUILayout.Label(label, GUILayout.Width(120));

			GUILayout.Label("Min", GUILayout.Width(30));
			data.Min = EditorGUILayout.FloatField(data.Min, GUILayout.Width(60));

			GUILayout.Label("Max", GUILayout.Width(30));
			data.Max = EditorGUILayout.FloatField(data.Max, GUILayout.Width(60));

			GUILayout.Label("W", GUILayout.Width(15));
			data.Weight = EditorGUILayout.Slider(data.Weight, 0f, 1f);

			EditorGUILayout.EndHorizontal();
		}

		private void DrawMinMax(string label, FloatMinMax data) {
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField(label, GUILayout.Width(150));

			data.Min = EditorGUILayout.FloatField("Min", data.Min);
			data.Max = EditorGUILayout.FloatField("Max", data.Max);

			EditorGUILayout.EndHorizontal();
		}

		private void DrawSaveSection() {
			EditorGUILayout.LabelField("=== SAVE CONFIG ===", EditorStyles.boldLabel);

			saveFileName = EditorGUILayout.TextField("File Name", saveFileName);

			if (GUILayout.Button("Save Pacing Config")) {
				SaveConfig();
			}
		}

		private void DrawLoadSection() {
			EditorGUILayout.LabelField("=== LOAD CONFIG ===", EditorStyles.boldLabel);

			//saveFileName = EditorGUILayout.TextField("File Name", saveFileName);
			if (GUILayout.Button("Load Pacing Config")) {
				LoadConfig();
			}
		}

		private void SaveConfig() {
			PacingTargetConfig config = new PacingTargetConfig();

			config.ThreatTargets = new List<float>(threats);
			config.TempoTargets = new List<float>(tempos);
			config.GlobalConstraints = constraintConfig;

			string json = JsonUtility.ToJson(config, true);

			string path = EditorUtility.SaveFilePanel(
				"Save Pacing Config",
				Application.dataPath,
				saveFileName,
				"json");

			if (!string.IsNullOrEmpty(path)) {
				System.IO.File.WriteAllText(path, json);
				Debug.Log("Pacing config saved to: " + path);
				AssetDatabase.Refresh();
			}
		}

		private void LoadConfig() {
			string path = EditorUtility.OpenFilePanel(
				"Load Pacing Config",
				Application.dataPath,
				"json");

			if (string.IsNullOrEmpty(path))
				return;

			string json = System.IO.File.ReadAllText(path);
			PacingTargetConfig config = JsonUtility.FromJson<PacingTargetConfig>(json);

			if (config != null) {
				threats = new List<float>(config.ThreatTargets);
				tempos = new List<float>(config.TempoTargets);

				overall.Clear();
				for (int i = 0; i < threats.Count; i++)
					overall.Add((threats[i] + tempos[i]) * 0.5f);

				constraintConfig = config.GlobalConstraints;

				Debug.Log("Pacing config loaded.");
			}
		}

		private void DrawGraphSection() {
			EditorGUILayout.LabelField("=== TARGET PACING ===", EditorStyles.boldLabel);
			EditorGUILayout.HelpBox("\"Threat & Tempo curves are editable via graph or fields below.", MessageType.Info);

			Rect rect = GUILayoutUtility.GetRect(position.width - 20, 400);

			EditorGUI.DrawRect(rect, new Color(0.12f, 0.12f, 0.12f));

			DrawGrid(rect);
			DrawAxes(rect);
			DrawAndEdit(rect);

			//// Update overall automatically
			//for (int i = 0; i < threats.Count; i++)
			//	overall[i] = (threats[i] + tempos[i]) * 0.5f;
			UpdateOverall();

			DrawLegendInsideGraph(rect);
			EditorGUILayout.Space(20);

			DrawSegmentFields();
		}

		private void DrawLegendInsideGraph(Rect rect) {
			float boxWidth = 120f;
			float boxHeight = 70f;
			float margin = 10f;

			Rect legendRect = new Rect(
				rect.xMax - boxWidth - margin,
				rect.y + margin,
				boxWidth,
				boxHeight
			);

			// Background
			EditorGUI.DrawRect(legendRect, new Color(0f, 0f, 0f, 0.6f));

			float lineHeight = 20f;
			float colorSize = 12f;

			DrawLegendItemInRect(
				new Rect(legendRect.x + 10, legendRect.y + 10, boxWidth - 20, lineHeight),
				Color.red,
				"Threat");

			DrawLegendItemInRect(
				new Rect(legendRect.x + 10, legendRect.y + 30, boxWidth - 20, lineHeight),
				Color.cyan,
				"Tempo");

			DrawLegendItemInRect(
				new Rect(legendRect.x + 10, legendRect.y + 50, boxWidth - 20, lineHeight),
				Color.green,
				"Overall");
		}

		private void DrawLegendItemInRect(Rect rowRect, Color color, string label) {
			float colorSize = 12f;

			Rect colorRect = new Rect(
				rowRect.x,
				rowRect.y + 4,
				colorSize,
				colorSize
			);

			EditorGUI.DrawRect(colorRect, color);

			EditorGUI.LabelField(
				new Rect(rowRect.x + colorSize + 8, rowRect.y, rowRect.width, rowRect.height),
				label,
				EditorStyles.whiteLabel
			);
		}

		private void DrawSegmentFields() {
			showSegmentFields = EditorGUILayout.Foldout(showSegmentFields, "Segment Target Values", true);

			if (!showSegmentFields)
				return;

			EditorGUILayout.BeginVertical("box");

			for (int i = 0; i < DATA_COUNT; i++) {
				EditorGUILayout.BeginHorizontal();

				GUILayout.Label($"Seg {i}", GUILayout.Width(60));

				threats[i] = EditorGUILayout.Slider(threats[i], 0f, 1f);
				tempos[i] = EditorGUILayout.Slider(tempos[i], 0f, 1f);

				EditorGUILayout.EndHorizontal();
			}

			EditorGUILayout.EndVertical();
		}

		private void UpdateOverall() {
			if (overall.Count != threats.Count)
				overall = new List<float>(new float[threats.Count]);

			for (int i = 0; i < threats.Count; i++)
				overall[i] = (threats[i] + tempos[i]) * 0.5f;
		}
	}

	[Serializable]
	public class PacingTargetConfig
	{
		public List<float> ThreatTargets = new();
		public List<float> TempoTargets = new();

		public ConstraintConfig GlobalConstraints = new ConstraintConfig();
	}

	[Serializable]
	public class ConstraintConfig
	{
		public FloatMinMax CollisionRatio = new();
		public FloatMinMax AbilityRatio = new();
		public FloatMinMax Angle = new();
		public FloatMinMax SafeDistance = new();

		public FloatMinMax ActionIntensity = new();
		public FloatMinMax ActionDensity = new();
		public FloatMinMax BotsDistance = new();
		public FloatMinMax Velocity = new();
	}

	[Serializable]
	public class FloatMinMax
	{
		public float Min;
		public float Max;

		[Range(0f, 1f)]
		public float Weight = 1;
	}

}
