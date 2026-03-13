using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace PacingFramework
{
	public class PacingEditorWindow : EditorWindow
	{
		#region Fields

		private PacingModel model;
		private GraphRenderer graphRenderer;
		private ConstraintDrawer constraintDrawer;

		private Vector2 scrollPos;
		private Vector2 segmentScrollPos;
		private float globalTolerance = 0.1f;

		private bool showConstraints = true;
		private bool showSegments = true;
		private bool isDirty;
		private string configPath = "";

		// [Todo] Get this from battle info. 
		private float battleTimer = 60f;
		private int segmentCount = 25;
		private int selectedSegmentTimeIndex = 0;
		private readonly int[] segmentTimeOptions = new int[] { 1, 5, 10 };
		private readonly string[] segmentTimeLabels = new string[] { "1 Second", "5 Seconds", "10 Seconds" };

		#endregion

		[MenuItem("Tools/Pacing Framework/Target Pacing")]
		public static void Open()
		{
			GetWindow<PacingEditorWindow>("Target Pacing");
		}

		private void OnEnable()
		{
			model = new PacingModel(segmentCount);
			graphRenderer = new GraphRenderer(model, () => MarkDirty());
			constraintDrawer = new ConstraintDrawer(model, () => MarkDirty());
		}

		private void MarkDirty()
		{
			isDirty = true;
			Repaint();
		}

		private void OnGUI()
		{
			titleContent.text = "Target Pacing Editor" + (isDirty ? " *" : "");

			scrollPos = EditorGUILayout.BeginScrollView(scrollPos);

			//DrawSegmentCountField();
			DrawSegmentSettings();
			graphRenderer.Draw(position.width - 20);

			EditorGUILayout.Space(10);
			showSegments = EditorGUILayout.Foldout(showSegments, "Target Segment Pacings", true, EditorStyles.foldoutHeader);

			if (showSegments)
				DrawSegmentFields();

			EditorGUILayout.Space(10);
			showConstraints = EditorGUILayout.Foldout(showConstraints, "Global Constraints", true, EditorStyles.foldoutHeader);

			if (showConstraints)
				constraintDrawer.Draw();

			EditorGUILayout.Space(10);
			DrawSaveLoadSection();

			EditorGUILayout.EndScrollView();
		}

		//private void DrawSegmentCountField() {
		//	EditorGUI.BeginChangeCheck();
		//	int newCount = EditorGUILayout.IntField("Segment Count", segmentCount);

		//	if (EditorGUI.EndChangeCheck()) {
		//		newCount = Mathf.Max(2, newCount);
		//		segmentCount = newCount;
		//		model.Resize(segmentCount);
		//		MarkDirty();
		//	}
		//}

		private void DrawSegmentSettings()
		{
			EditorGUI.BeginChangeCheck();

			EditorGUILayout.BeginHorizontal();

			// Dropdown
			selectedSegmentTimeIndex = EditorGUILayout.Popup(
				"Segment Time",
				selectedSegmentTimeIndex,
				segmentTimeLabels
			);

			int selectedSegmentTime = segmentTimeOptions[selectedSegmentTimeIndex];

			GUILayout.Space(10);
			// Calculate segment count
			int calculatedSegmentCount = Mathf.Max(2, Mathf.FloorToInt(battleTimer / selectedSegmentTime));

			// Read-only Segment Count field
			GUI.enabled = false;
			EditorGUILayout.IntField("Segment Count", calculatedSegmentCount);
			GUI.enabled = true;

			EditorGUILayout.EndHorizontal();

			if (EditorGUI.EndChangeCheck())
			{
				segmentCount = calculatedSegmentCount;

				if (model != null)
				{
					model.Resize(segmentCount);
				}

				MarkDirty();
			}
		}

		private void DrawSegmentFields()
		{
			EditorGUILayout.BeginVertical("box");

			EditorGUI.BeginChangeCheck();
			globalTolerance = EditorGUILayout.Slider(
				"Global Tolerance",
				globalTolerance,
				0f,
				1f
			);
			if (EditorGUI.EndChangeCheck())
			{
				globalTolerance = Mathf.Max(0f, globalTolerance);
				MarkDirty();
			}

			EditorGUILayout.Space(5);

			int visibleCount = Mathf.Min(model.Count, 10);
			float rowHeight = EditorGUIUtility.singleLineHeight + 10f;
			float scrollHeight = visibleCount * rowHeight;

			segmentScrollPos = EditorGUILayout.BeginScrollView(
				segmentScrollPos,
				GUILayout.Height(scrollHeight)
			);

			for (int i = 0; i < model.Count; i++)
			{
				DrawSegmentRow(i);
			}

			EditorGUILayout.EndScrollView();
			EditorGUILayout.EndVertical();
		}

		private void DrawSegmentRow(int i)
		{
			const float labelWidth = 60f;
			const float floatWidth = 50f;

			EditorGUILayout.BeginHorizontal("box");

			GUILayout.Label($"Seg {i}", GUILayout.Width(55));

			DrawMetricHorizontal(
				"Threat",
				model.Threats,
				model.SetThreat,
				i,
				labelWidth,
				floatWidth
			);

			GUILayout.Space(10);

			DrawMetricHorizontal(
				"Tempo",
				model.Tempos,
				model.SetTempo,
				i,
				labelWidth,
				floatWidth
			);

			EditorGUILayout.EndHorizontal();
		}

		private float Round2(float v)
		{
			return Mathf.Round(v * 100f) / 100f;
		}

		private void DrawMetricHorizontal(string label, List<float> values, System.Action<int, float> setter, int index, float labelWidth, float floatWidth)
		{
			float half = Round2(globalTolerance * 0.5f);
			float target = Round2(values[index]);

			float min = Round2(Mathf.Clamp01(target - half));
			float max = Round2(Mathf.Clamp01(target + half));

			float sliderMin = min;
			float sliderMax = max;

			GUILayout.Label(label, GUILayout.Width(labelWidth));

			// MIN FIELD
			EditorGUI.BeginChangeCheck();
			//float newMin = EditorGUILayout.FloatField(min, GUILayout.Width(floatWidth));
			float newMin = EditorGUILayout.FloatField(Round2(min), GUILayout.Width(floatWidth));
			bool minChanged = EditorGUI.EndChangeCheck();

			// SLIDER
			EditorGUI.BeginChangeCheck();
			EditorGUILayout.MinMaxSlider(ref sliderMin, ref sliderMax, 0f, 1f);
			bool sliderChanged = EditorGUI.EndChangeCheck();

			// MAX FIELD
			EditorGUI.BeginChangeCheck();
			//float newMax = EditorGUILayout.FloatField(max, GUILayout.Width(floatWidth));
			float newMax = EditorGUILayout.FloatField(Round2(max), GUILayout.Width(floatWidth));
			bool maxChanged = EditorGUI.EndChangeCheck();

			// APPLY CHANGES
			if (sliderChanged)
			{
				float newTarget = Round2((sliderMin + sliderMax) * 0.5f);
				setter(index, Mathf.Clamp01(newTarget));
				MarkDirty();
			}
			else if (minChanged)
			{
				float newTarget = Round2(newMin + half);
				setter(index, Mathf.Clamp01(newTarget));
				MarkDirty();
			}
			else if (maxChanged)
			{
				float newTarget = Round2(newMax - half);
				setter(index, Mathf.Clamp01(newTarget));
				MarkDirty();
			}
		}

		#region Save Load

		private void DrawSaveLoadSection()
		{
			EditorGUILayout.LabelField("Pacing Config", EditorStyles.boldLabel);

			EditorGUI.BeginDisabledGroup(true);
			EditorGUILayout.TextField("File Path", configPath);
			EditorGUI.EndDisabledGroup();

			EditorGUILayout.BeginHorizontal();

			if (GUILayout.Button("Save", GUILayout.Height(30)))
				SaveConfig();

			if (GUILayout.Button("Load", GUILayout.Height(30)))
				LoadConfig();

			EditorGUILayout.EndHorizontal();
		}

		private void SaveConfig()
		{
            string path;
            if (string.IsNullOrEmpty(configPath))
				path = EditorUtility.SaveFilePanel("Save Config", "Assets", "PacingConfig", "json");
			else
				path = configPath;

			if (string.IsNullOrEmpty(path)) return;

			var config = model.ToConfig();
			string json = JsonUtility.ToJson(config, true);
			System.IO.File.WriteAllText(path, json);

			configPath = path;
			isDirty = false;
		}

		private void LoadConfig()
		{
			string path = EditorUtility.OpenFilePanel("Load Config", "Assets", "json");
			if (string.IsNullOrEmpty(path)) return;

			string json = System.IO.File.ReadAllText(path);
			var config = JsonUtility.FromJson<PacingTargetConfig>(json);

			model.FromConfig(config);
			segmentCount = model.Count;

			configPath = path;
			isDirty = false;

			Repaint();
		}

		#endregion
	}

	#region MODEL

	[Serializable]
	public class PacingModel
	{
		public List<float> Threats = new();
		public List<float> Tempos = new();
		public List<float> Overall = new();

		public ConstraintConfig Constraints = new();

		public int Count => Threats.Count;

		public PacingModel(int count)
		{
			Resize(count);
		}

		public void Resize(int count)
		{
			Threats = ResizeList(Threats, count);
			Tempos = ResizeList(Tempos, count);
			Overall = ResizeList(Overall, count);
			UpdateOverall();
		}

		private List<float> ResizeList(List<float> list, int count)
		{
			var newList = new List<float>(count);

			for (int i = 0; i < count; i++)
			{
				if (i < list.Count)
					newList.Add(list[i]);
				else
					newList.Add(0.5f);
			}

			return newList;
		}

		public void SetThreat(int i, float v)
		{
			Threats[i] = Round2(v);
			UpdateOverall();
		}

		public void SetTempo(int i, float v)
		{
			Tempos[i] = Round2(v);
			UpdateOverall();
		}

		private void UpdateOverall()
		{
			for (int i = 0; i < Count; i++)
				Overall[i] = (Threats[i] + Tempos[i]) * 0.5f;
		}

		private float Round2(float v) => Mathf.Round(v * 100f) / 100f;

		public PacingTargetConfig ToConfig()
		{
			return new PacingTargetConfig
			{
				ThreatTargets = new List<float>(Threats),
				TempoTargets = new List<float>(Tempos),
				GlobalConstraints = Constraints
			};
		}

		public void FromConfig(PacingTargetConfig config)
		{
			Threats = new List<float>(config.ThreatTargets);
			Tempos = new List<float>(config.TempoTargets);
			Resize(Threats.Count);
			Constraints = config.GlobalConstraints;
		}
	}

	#endregion

	#region GRAPH RENDERER

	public class GraphRenderer
	{
		private PacingModel model;
		private Action onChanged;

		private const float padding = 50f;
		private const float pointRadius = 5f;

		private int draggingCurve = -1;
		private int draggingIndex = -1;

		public GraphRenderer(PacingModel model, Action onChanged)
		{
			this.model = model;
			this.onChanged = onChanged;
		}

		public void Draw(float width)
		{
			Rect rect = GUILayoutUtility.GetRect(width, 400);
			EditorGUI.DrawRect(rect, new Color(0.12f, 0.12f, 0.12f));

			DrawGrid(rect);
			DrawAxes(rect);

			DrawCurve(rect, model.Threats, Color.red, 0);
			DrawCurve(rect, model.Tempos, Color.cyan, 1);
			DrawCurve(rect, model.Overall, Color.green, -1); // visual only (but with circles)

			DrawLegend(rect);
		}

		private void DrawGrid(Rect rect)
		{
			Handles.BeginGUI();
			Handles.color = new Color(0.3f, 0.3f, 0.3f, 0.4f);

			float left = rect.x + padding;
			float right = rect.x + rect.width - padding;
			float top = rect.y + padding;
			float bottom = rect.y + rect.height - padding;

			float width = right - left;
			float height = bottom - top;

			int xSteps = model.Count - 1;

			for (int i = 0; i <= xSteps; i++)
			{
				float x = left + i / (float)xSteps * width;
				Handles.DrawLine(new Vector3(x, top), new Vector3(x, bottom));

				if (i % 5 == 0 || i == 0 || i == xSteps)
					GUI.Label(new Rect(x - 10, bottom + 2, 40, 20), i.ToString());
			}

			for (int i = 0; i <= 4; i++)
			{
				float t = i / 4f;
				float y = bottom - t * height;
				Handles.DrawLine(new Vector3(left, y), new Vector3(right, y));

				GUI.Label(new Rect(left - 35, y - 8, 30, 20), t.ToString("0.##"));
			}

			Handles.EndGUI();
		}

		private void DrawAxes(Rect rect)
		{
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

		private void DrawCurve(Rect rect, List<float> list, Color color, int curveIndex)
		{
			Handles.BeginGUI();
			Handles.color = color;

			float left = rect.x + padding;
			float right = rect.x + rect.width - padding;
			float top = rect.y + padding;
			float bottom = rect.y + rect.height - padding;

			float width = right - left;
			float height = bottom - top;

			Event e = Event.current;

			Vector2 GetPoint(int i)
			{
				float x = left + i / (float)(list.Count - 1) * width;
				float y = bottom - Mathf.Clamp01(list[i]) * height;
				return new Vector2(x, y);
			}

			// Draw lines
			for (int i = 0; i < list.Count - 1; i++)
				Handles.DrawLine(GetPoint(i), GetPoint(i + 1));

			// Draw circles (including overall)
			for (int i = 0; i < list.Count; i++)
			{
				Vector2 p = GetPoint(i);
				Handles.DrawSolidDisc(p, Vector3.forward, pointRadius);

				if (curveIndex < 0)
					continue; // overall not draggable

				if (e.type == EventType.MouseDown &&
					Vector2.Distance(e.mousePosition, p) < pointRadius)
				{
					draggingCurve = curveIndex;
					draggingIndex = i;
					e.Use();
				}

				if (e.type == EventType.MouseDrag &&
					draggingCurve == curveIndex &&
					draggingIndex == i)
				{
					float normalized = Mathf.Clamp01((bottom - e.mousePosition.y) / height);

					if (curveIndex == 0)
						model.SetThreat(i, normalized);
					else
						model.SetTempo(i, normalized);

					onChanged?.Invoke();
					e.Use();
				}
			}

			if (e.type == EventType.MouseUp)
			{
				draggingCurve = -1;
				draggingIndex = -1;
			}

			Handles.EndGUI();
		}

		private void DrawLegend(Rect rect)
		{
			float boxWidth = 95f;
			float boxHeight = 75f;
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
		}

		private void DrawLegendItem(Rect legendRect, int row, Color color, string label)
		{
			float rowHeight = 22f;
			float y = legendRect.y + 8 + row * rowHeight;

			Rect colorRect = new Rect(legendRect.x + 8, y + 4, 12, 12);
			EditorGUI.DrawRect(colorRect, color);

			EditorGUI.LabelField(
				new Rect(legendRect.x + 25, y, 60, 20),
				label,
				EditorStyles.whiteLabel
			);
		}
	}

	#endregion

	#region CONSTRAINT DRAWER

	public class ConstraintDrawer
	{
		private PacingModel model;
		private Action onChanged;

		public ConstraintDrawer(PacingModel model, Action onChanged)
		{
			this.model = model;
			this.onChanged = onChanged;
		}

		public void Draw()
		{
			EditorGUILayout.BeginVertical("box");

			DrawConstraintRow("Collision Ratio", model.Constraints.CollisionRatio);
			DrawConstraintRow("Ability Ratio", model.Constraints.AbilityRatio);
			DrawConstraintRow("Angle", model.Constraints.Angle);
			DrawConstraintRow("Safe Distance", model.Constraints.SafeDistance);

			EditorGUILayout.Space();

			DrawConstraintRow("Action Intensity", model.Constraints.ActionIntensity);
			DrawConstraintRow("Action Density", model.Constraints.ActionDensity);
			DrawConstraintRow("Bots Distance", model.Constraints.BotsDistance);
			DrawConstraintRow("Velocity", model.Constraints.Velocity);

			EditorGUILayout.EndVertical();
		}

		private void DrawConstraintRow(string label, ConstraintMinMax data)
		{
			EditorGUILayout.BeginHorizontal();

			GUILayout.Label(label, GUILayout.Width(130));

			// ----- MinLimit -----
			GUILayout.Label("MinL", GUILayout.Width(35));
			EditorGUI.BeginChangeCheck();
			float newMinLimit = EditorGUILayout.FloatField(data.MinLimit, GUILayout.Width(60));
			bool minLimitChanged = EditorGUI.EndChangeCheck();

			// ----- Min -----
			GUILayout.Label("Min", GUILayout.Width(30));
			EditorGUI.BeginChangeCheck();
			float newMin = EditorGUILayout.DelayedFloatField(data.Min, GUILayout.Width(60));
			bool minFieldChanged = EditorGUI.EndChangeCheck();

			// ----- Slider -----
			float sliderMin = data.Min;
			float sliderMax = data.Max;

			EditorGUI.BeginChangeCheck();
			EditorGUILayout.MinMaxSlider(ref sliderMin, ref sliderMax, data.MinLimit, data.MaxLimit);
			bool sliderChanged = EditorGUI.EndChangeCheck();

			// ----- Max -----
			GUILayout.Label("Max", GUILayout.Width(30));
			EditorGUI.BeginChangeCheck();
			float newMax = EditorGUILayout.DelayedFloatField(data.Max, GUILayout.Width(60));
			bool maxFieldChanged = EditorGUI.EndChangeCheck();

			// ----- MaxLimit -----
			GUILayout.Label("MaxL", GUILayout.Width(35));
			EditorGUI.BeginChangeCheck();
			float newMaxLimit = EditorGUILayout.FloatField(data.MaxLimit, GUILayout.Width(60));
			bool maxLimitChanged = EditorGUI.EndChangeCheck();

			// ----- Weight -----
			GUILayout.Label("W", GUILayout.Width(20));
			EditorGUI.BeginChangeCheck();
			float newWeight = EditorGUILayout.Slider(data.Weight, 0f, 1f);
			bool weightChanged = EditorGUI.EndChangeCheck();

			EditorGUILayout.EndHorizontal();

			bool changed = false;

			// ----- Apply only the control that changed -----
			// ----- MinLimit -----
			if (minLimitChanged)
			{
				float r = Round2(newMinLimit);
				if (!NearlyEqual(r, data.MinLimit))
				{
					data.MinLimit = r;
					changed = true;
				}
			}

			// ----- MaxLimit -----
			if (maxLimitChanged)
			{
				float r = Round2(newMaxLimit);
				if (!NearlyEqual(r, data.MaxLimit))
				{
					data.MaxLimit = r;
					changed = true;
				}
			}

			// ----- Slider -----
			if (sliderChanged)
			{
				float rMin = Round2(sliderMin);
				float rMax = Round2(sliderMax);

				if (!NearlyEqual(rMin, data.Min))
				{
					data.Min = rMin;
					changed = true;
				}

				if (!NearlyEqual(rMax, data.Max))
				{
					data.Max = rMax;
					changed = true;
				}

				GUI.FocusControl(null);
				EditorGUIUtility.editingTextField = false;
			}

			// ----- Min Field -----
			if (minFieldChanged)
			{
				float r = Round2(newMin);
				if (!NearlyEqual(r, data.Min))
				{
					data.Min = r;
					changed = true;
				}
			}

			// ----- Max Field -----
			if (maxFieldChanged)
			{
				float r = Round2(newMax);
				if (!NearlyEqual(r, data.Max))
				{
					data.Max = r;
					changed = true;
				}
			}

			// ----- Weight -----
			if (weightChanged)
			{
				float r = Round2(newWeight);
				if (!NearlyEqual(r, data.Weight))
				{
					data.Weight = r;
					changed = true;
				}
			}

			if (changed)
			{
				Validate(data);
				onChanged?.Invoke();
			}
		}

		private void Validate(ConstraintMinMax c)
		{
			if (c.MinLimit > c.MaxLimit)
				c.MaxLimit = c.MinLimit;

			c.Min = Mathf.Clamp(c.Min, c.MinLimit, c.MaxLimit);
			c.Max = Mathf.Clamp(c.Max, c.MinLimit, c.MaxLimit);

			if (c.Min > c.Max)
				c.Min = c.Max;
		}

		private float Round2(float v) => Mathf.Round(v * 100f) / 100f;

		private bool NearlyEqual(float a, float b)
		{
			float EPS = 0.0001f;
			return Mathf.Abs(a - b) < EPS;
		}
	}
	#endregion
}