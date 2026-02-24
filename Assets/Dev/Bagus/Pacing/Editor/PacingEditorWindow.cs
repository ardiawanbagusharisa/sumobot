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
		private bool isDirty = false;

		private ConstraintConfig constraintConfig = new ConstraintConfig();
		private Vector2 scrollPos;
		private string configPath = "";

		private const int DATA_COUNT = 25;
		private const float padding = 50f;
		private const float pointSize = 10f;

		private List<float> threats = new List<float>();
		private List<float> tempos = new List<float>();
		private List<float> overall = new List<float>();

		private int draggingCurve = -1;
		private int draggingIndex = -1;

		[MenuItem("Tools/Pacing Framework/Target Pacing Editor")]
		public static void Open() {
			GetWindow<PacingEditorWindow>("Target Pacing Editor");
		}

		private void OnEnable() {
			isDirty = false;
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

				threats.Add(Round2(threat));
				tempos.Add(Round2(tempo));
				overall.Add(Round2(over));
			}
		}

		private bool hasInitialized = false;

		private void OnGUI() {
			// mark first GUI frame as initialized
			if (!hasInitialized) {
				hasInitialized = true;
				isDirty = false;
			}

			// show * in title
			titleContent.text = "Target Pacing Editor" + (isDirty ? " *" : "");

			scrollPos = EditorGUILayout.BeginScrollView(scrollPos);
			DrawTargetPacingSection();
			EditorGUILayout.EndScrollView();

			Repaint();
		}

		private void DrawTargetPacingSection() {
			DrawGraphSection();
			EditorGUILayout.Space(10);

			DrawConstraintSection();
			EditorGUILayout.Space(10);

			DrawSaveLoadSection();
			EditorGUILayout.Space(10);
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
						list[i] = Round2(normalized);
						isDirty = true;
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
			GUIStyle boldFoldout = new GUIStyle(EditorStyles.foldout) { fontStyle = FontStyle.Bold };
			showConstraints = EditorGUILayout.Foldout(showConstraints, "Global Constraints", true, boldFoldout);

			if (!showConstraints)
				return;

			EditorGUILayout.BeginVertical("box");

			DrawConstraintRow("Collision Ratio", constraintConfig.CollisionRatio);
			DrawConstraintRow("Ability Ratio", constraintConfig.AbilityRatio);
			DrawConstraintRow("Angle", constraintConfig.Angle);
			DrawConstraintRow("Safe Distance", constraintConfig.SafeDistance);

			EditorGUILayout.Space(5);

			DrawConstraintRow("Action Intensity", constraintConfig.ActionIntensity);
			DrawConstraintRow("Action Density", constraintConfig.ActionDensity);
			DrawConstraintRow("Bots Distance", constraintConfig.BotsDistance);
			DrawConstraintRow("Velocity", constraintConfig.Velocity);

			EditorGUILayout.EndVertical();
		}

		private void DrawConstraintRow(string label, ConstraintMinMax data) {
			EditorGUILayout.BeginHorizontal();

			GUILayout.Label(label, GUILayout.Width(130));

			// ----- MinLimit -----
			GUILayout.Label("MinL", GUILayout.Width(35));
			EditorGUI.BeginChangeCheck();
			float newMinLimit = EditorGUILayout.FloatField(
				data.MinLimit,
				GUILayout.Width(60));
			bool minLimitChanged = EditorGUI.EndChangeCheck();

			// ----- Min -----
			GUILayout.Label("Min", GUILayout.Width(28));
			EditorGUI.BeginChangeCheck();
			float newMin = EditorGUILayout.DelayedFloatField(
				data.Min,
				GUILayout.Width(60));
			bool minChanged = EditorGUI.EndChangeCheck();

			// ----- Slider -----
			EditorGUI.BeginChangeCheck();
			EditorGUILayout.MinMaxSlider(
				ref data.Min,
				ref data.Max,
				data.MinLimit,
				data.MaxLimit
			);
			bool sliderChanged = EditorGUI.EndChangeCheck();

			// ----- Max -----
			GUILayout.Label("Max", GUILayout.Width(30));
			EditorGUI.BeginChangeCheck();
			float newMax = EditorGUILayout.DelayedFloatField(
				data.Max,
				GUILayout.Width(60));
			bool maxChanged = EditorGUI.EndChangeCheck();

			// ----- MaxLimit -----
			GUILayout.Label("MaxL", GUILayout.Width(35));
			EditorGUI.BeginChangeCheck();
			float newMaxLimit = EditorGUILayout.FloatField(
				data.MaxLimit,
				GUILayout.Width(60));
			bool maxLimitChanged = EditorGUI.EndChangeCheck();

			// ----- Weight -----
			GUILayout.Label("W", GUILayout.Width(18));
			EditorGUI.BeginChangeCheck();
			float newWeight = EditorGUILayout.Slider(data.Weight, 0f, 1f);
			bool weightChanged = EditorGUI.EndChangeCheck();

			EditorGUILayout.EndHorizontal();

			// ---------- Apply Changes ----------

			if (minLimitChanged) {
				data.MinLimit = Round2(newMinLimit);
				isDirty = true;
			}

			if (maxLimitChanged) {
				data.MaxLimit = Round2(newMaxLimit);
				isDirty = true;
			}

			if (minChanged) {
				data.Min = Round2(newMin);
				isDirty = true;
			}

			if (maxChanged) {
				data.Max = Round2(newMax);
				isDirty = true;
			}

			if (sliderChanged) {
				data.Min = Round2(data.Min);
				data.Max = Round2(data.Max);

				// Release text field focus so value refreshes
				GUI.FocusControl(null);
				EditorGUIUtility.editingTextField = false;

				isDirty = true;
			}

			if (weightChanged) {
				data.Weight = Round2(newWeight);
				isDirty = true;
			}

			// ---------- Validation ----------
			// --- Detect MaxLimit change ---
			if (!Mathf.Approximately(data.PreviousMaxLimit, data.MaxLimit)) {
				// If max was at old limit, keep it attached
				if (Mathf.Approximately(data.Max, data.PreviousMaxLimit))
					data.Max = data.MaxLimit;

				data.PreviousMaxLimit = data.MaxLimit;
			}

			// --- Detect MinLimit change ---
			if (!Mathf.Approximately(data.PreviousMinLimit, data.MinLimit)) {
				if (Mathf.Approximately(data.Min, data.PreviousMinLimit))
					data.Min = data.MinLimit;

				data.PreviousMinLimit = data.MinLimit;
			}

			// --- Safety ---
			if (data.MinLimit > data.MaxLimit)
				data.MaxLimit = data.MinLimit;

			if (data.Max > data.MaxLimit)
				data.Max = data.MaxLimit;

			if (data.Min < data.MinLimit)
				data.Min = data.MinLimit;

			if (data.Min > data.Max)
				data.Min = data.Max;

		}

		private void DrawSaveLoadSection() {
			EditorGUILayout.LabelField("Pacing Config", EditorStyles.boldLabel);

			EditorGUI.BeginDisabledGroup(true);
			configPath = EditorGUILayout.TextField("File Path", configPath);
			EditorGUI.EndDisabledGroup();
			EditorGUILayout.BeginHorizontal();

			if (GUILayout.Button("Save", GUILayout.Height(30))) {
				SaveConfig(configPath);
			}

			if (GUILayout.Button("Load", GUILayout.Height(30))) {
				LoadConfig(configPath);
			}

			EditorGUILayout.EndHorizontal();
		}

		private string ToRelativePath(string absolutePath) {
			if (absolutePath.StartsWith(Application.dataPath))
				return "Assets" + absolutePath.Substring(Application.dataPath.Length);
			return absolutePath; // outside project
		}

		// Convert relative path to absolute
		private string ToAbsolutePath(string relativePath) {
			if (relativePath.StartsWith("Assets"))
				return Application.dataPath + relativePath.Substring("Assets".Length);
			return relativePath; // outside project
		}

		private void SaveConfig(string currentPath) {
			PacingTargetConfig config = new PacingTargetConfig {
				ThreatTargets = new List<float>(threats),
				TempoTargets = new List<float>(tempos),
				GlobalConstraints = constraintConfig
			};

			string json = JsonUtility.ToJson(config, true);

			string defaultFolder = "Assets/Dev/Bagus/Pacing";
			string absolutePath = EditorUtility.SaveFilePanel(
				"Save Pacing Config",
				defaultFolder,
				"PacingConfig",
				"json");

			if (string.IsNullOrEmpty(absolutePath))
				return;

			System.IO.File.WriteAllText(absolutePath, json);
			AssetDatabase.Refresh();

			configPath = ToRelativePath(absolutePath); // store as relative
			isDirty = false; // just saved
			Debug.Log("Saved to: " + configPath);
		}

		private void LoadConfig(string currentPath) {
			if (isDirty) {
				if (!EditorUtility.DisplayDialog("Unsaved Changes",
						"You have unsaved changes. Do you want to continue loading and lose changes?",
						"Yes", "Cancel")) {
					return;
				}
			}

			string defaultFolder = "Assets/Dev/Bagus/Pacing";
			string absolutePath = EditorUtility.OpenFilePanel(
				"Load Pacing Config",
				defaultFolder,
				"json");

			if (string.IsNullOrEmpty(absolutePath))
				return;

			if (!System.IO.File.Exists(absolutePath)) {
				Debug.LogWarning("File not found.");
				return;
			}

			string json = System.IO.File.ReadAllText(absolutePath);
			PacingTargetConfig config = JsonUtility.FromJson<PacingTargetConfig>(json);

			if (config == null) {
				Debug.LogWarning("Invalid config file.");
				return;
			}

			threats = new List<float>(config.ThreatTargets);
			tempos = new List<float>(config.TempoTargets);
			for (int i = 0; i < threats.Count; i++) {
				threats[i] = Round2(threats[i]);
				tempos[i] = Round2(tempos[i]);
			}
			constraintConfig = config.GlobalConstraints;
			NormalizeConstraints(constraintConfig);
			ValidateConstraintLimits(constraintConfig);

			UpdateOverall();

			configPath = ToRelativePath(absolutePath);
			isDirty = false; // just loaded

			Repaint();

			Debug.Log("Loaded from: " + configPath);
		}

		private void NormalizeConstraints(ConstraintConfig config) {
			void Normalize(ConstraintMinMax c) {
				c.Min = Round2(c.Min);
				c.Max = Round2(c.Max);
				c.Weight = Round2(c.Weight);
			}

			Normalize(config.CollisionRatio);
			Normalize(config.AbilityRatio);
			Normalize(config.Angle);
			Normalize(config.SafeDistance);
			Normalize(config.ActionIntensity);
			Normalize(config.ActionDensity);
			Normalize(config.BotsDistance);
			Normalize(config.Velocity);
		}

		private void ValidateConstraintLimits(ConstraintConfig config) {
			void Validate(ConstraintMinMax c) {
				// Ensure limits are valid
				if (c.MinLimit > c.MaxLimit)
					c.MaxLimit = c.MinLimit;

				// Clamp values inside limits
				c.Min = Mathf.Clamp(c.Min, c.MinLimit, c.MaxLimit);
				c.Max = Mathf.Clamp(c.Max, c.MinLimit, c.MaxLimit);

				if (c.Min > c.Max)
					c.Max = c.Min;

				// Force 2 decimal precision
				c.MinLimit = Round2(c.MinLimit);
				c.MaxLimit = Round2(c.MaxLimit);
				c.Min = Round2(c.Min);
				c.Max = Round2(c.Max);
				c.Weight = Round2(c.Weight);
			}

			Validate(config.CollisionRatio);
			Validate(config.AbilityRatio);
			Validate(config.Angle);
			Validate(config.SafeDistance);
			Validate(config.ActionIntensity);
			Validate(config.ActionDensity);
			Validate(config.BotsDistance);
			Validate(config.Velocity);
		}

		private void DrawGraphSection() {
			EditorGUILayout.LabelField("Target Pacing Curve", EditorStyles.boldLabel);
			EditorGUILayout.HelpBox("Edit segments' threat & tempo targets via graph or fields.", MessageType.Info);

			Rect rectCurveCanvas = GUILayoutUtility.GetRect(position.width - 20, 400);
			EditorGUI.DrawRect(rectCurveCanvas, new Color(0.12f, 0.12f, 0.12f));

			DrawGrid(rectCurveCanvas);
			DrawAxes(rectCurveCanvas);
			DrawAndEdit(rectCurveCanvas);
			DrawLegendInsideGraph(rectCurveCanvas);

			UpdateOverall();
			EditorGUILayout.Space(10);

			DrawSegmentFields();			
		}

		private void DrawLegendInsideGraph(Rect rect) {
			float boxWidth = 85f;
			float boxHeight = 70f;
			float margin = 10f;

			Rect legendRect = new Rect(
				rect.xMax - boxWidth - margin,
				rect.y + margin,
				boxWidth,
				boxHeight
			);

			// Background
			EditorGUI.DrawRect(legendRect, new Color(0f, 0f, 0f, 0.3f));

			float lineHeight = 20f;

			DrawLegendItemInRect(
				new Rect(legendRect.x + 10, legendRect.y + 5, boxWidth - 20, lineHeight),
				Color.red,
				"Threat");

			DrawLegendItemInRect(
				new Rect(legendRect.x + 10, legendRect.y + 25, boxWidth - 20, lineHeight),
				Color.cyan,
				"Tempo");

			DrawLegendItemInRect(
				new Rect(legendRect.x + 10, legendRect.y + 45, boxWidth - 20, lineHeight),
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
			GUIStyle boldFoldout = new GUIStyle(EditorStyles.foldout) {
				fontStyle = FontStyle.Bold
			};

			showSegmentFields = EditorGUILayout.Foldout(
				showSegmentFields,
				"Target Segment Pacings",
				true,
				boldFoldout);

			if (!showSegmentFields)
				return;

			EditorGUILayout.BeginVertical("box");

			for (int i = 0; i < DATA_COUNT; i++) {
				EditorGUILayout.BeginHorizontal();

				GUILayout.Label($"Segment {i}", GUILayout.Width(80));

				// Threat
				EditorGUI.BeginChangeCheck();
				float threatValue = EditorGUILayout.Slider(threats[i], 0f, 1f);
				if (EditorGUI.EndChangeCheck()) {
					threatValue = Mathf.Round(threatValue * 100f) / 100f;

					if (threatValue != threats[i]) {
						threats[i] = threatValue;
						isDirty = true;
					}
				}

				// Tempo
				EditorGUI.BeginChangeCheck();
				float tempoValue = EditorGUILayout.Slider(tempos[i], 0f, 1f);
				if (EditorGUI.EndChangeCheck()) {
					tempoValue = Mathf.Round(tempoValue * 100f) / 100f;

					if (tempoValue != tempos[i]) {
						tempos[i] = tempoValue;
						isDirty = true;
					}
				}

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

		private float Round2(float value) {
			return Mathf.Round(value * 100f) / 100f;
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
		// Default Constraints [Modify these constraints according to bot]. 
		public ConstraintMinMax CollisionRatio = new(0, 1);
		public ConstraintMinMax AbilityRatio = new(0, 0.2f);
		public ConstraintMinMax Angle = new(0, 180f);
		public ConstraintMinMax SafeDistance = new(1, 5);

		public ConstraintMinMax ActionIntensity = new(0, 50);
		public ConstraintMinMax ActionDensity = new(0, 1);
		public ConstraintMinMax BotsDistance = new(1, 5);
		public ConstraintMinMax Velocity = new(0, 10);
	}

	[Serializable]
	public class ConstraintMinMax
	{
		public float Min;
		public float Max;
		public float MinLimit;
		public float MaxLimit;

		[Range(0f, 1f)] 
		public float Weight = 1;
		[NonSerialized] 
		public float PreviousMaxLimit;
		[NonSerialized] 
		public float PreviousMinLimit;

		public ConstraintMinMax() {
			PreviousMinLimit = MinLimit = Min = 0f;
			PreviousMaxLimit = MaxLimit = Max = 1f;
		}

		public ConstraintMinMax(float minLimit, float maxLimit) {
			PreviousMinLimit = MinLimit = minLimit;
			PreviousMaxLimit = MaxLimit = maxLimit;
			Min = Mathf.Round(minLimit * 100f) / 100f;
			Max = Mathf.Round(maxLimit * 100f) / 100f;
		}

		public ConstraintMinMax(float minValue, float maxValue, float minLimit, float maxLimit) {
			PreviousMinLimit = MinLimit = minLimit;
			PreviousMaxLimit = MaxLimit = maxLimit;
			Min = minValue;
			Max = maxValue;
		}
	}

}
