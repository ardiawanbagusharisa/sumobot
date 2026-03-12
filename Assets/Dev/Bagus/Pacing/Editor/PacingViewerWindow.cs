using SumoBot;
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace PacingFramework
{
	public class PacingViewerWindow : EditorWindow
	{
		private bool showBots;
		private bool showPacingDetails;
		private bool showSegmentDetails;
		private bool showEvaluationDetails;

		private Vector2 pacingScroll;
		private Vector2 segmentScroll;
		private Vector2 evaluationScroll;

		private PacingController controller;
		private PacingTargetConfig targetConfig;
		private string loadedConfigPath = "";

		private Vector2 scroll;
		private const float padding = 50f;
		private const float pointRadius = 4f;

		private bool overlayTarget = true;

		// [Test] For testing only. 
		private string botName = "Bot1";
		private string botScore = "3";
		private string botName1 = "Bot2";
		private string botScore1 = "5";

		[MenuItem("Tools/Pacing Framework/Pacing Viewer")]
		public static void Open()
		{
			GetWindow<PacingViewerWindow>("Pacing Viewer");
		}

		private void OnGUI()
		{
			scroll = EditorGUILayout.BeginScrollView(scroll);

			DrawSelectionSection();

			if (controller == null)
			{
				EditorGUILayout.HelpBox("Assign a PacingController.", MessageType.Info);
				EditorGUILayout.EndScrollView();
				return;
			}

			GamePacing history = controller.GetHistory();

			if (history.SegmentPacings.Count == 0)
			{
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

			if (overlayTarget && targetConfig != null)
			{
				DrawTargetOverlay(rect, targetConfig, threat.Count);
				DrawEvaluation(threat, tempo);
			}

			DrawLegend(rect);

			DrawBotsSection(controller);
			DrawPacingDetails(history);
			DrawSegmentDetails(history);
			DrawSegmentEvaluation(threat, tempo);

			Repaint(); // live update

			EditorGUILayout.EndScrollView();
		}

		// ======================================================
		// UI
		// ======================================================

		private void DrawSelectionSection()
		{
			EditorGUILayout.BeginVertical("box");

			controller = (PacingController)EditorGUILayout.ObjectField(
				"Pacing Controller",
				controller,
				typeof(PacingController),
				true);

			EditorGUILayout.BeginVertical("box");

			EditorGUILayout.BeginHorizontal();

			EditorGUILayout.LabelField("Target Config (Override)", GUILayout.Width(160));

			EditorGUI.BeginDisabledGroup(true);
			EditorGUILayout.TextField("Loaded Path", loadedConfigPath);
			EditorGUI.EndDisabledGroup();

			if (GUILayout.Button("Load JSON", GUILayout.Width(100)))
			{
				string path = EditorUtility.OpenFilePanel(
					"Load Target Config",
					"",
					"json");

				if (!string.IsNullOrEmpty(path))
				{
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

		private List<float> ExtractThreat(GamePacing history)
		{
			var list = new List<float>();
			foreach (var p in history.SegmentPacings)
				list.Add(p.Threat.Value);
			return list;
		}

		private List<float> ExtractTempo(GamePacing history)
		{
			var list = new List<float>();
			foreach (var p in history.SegmentPacings)
				list.Add(p.Tempo.Value);
			return list;
		}

		private List<float> ExtractOverall(GamePacing history)
		{
			var list = new List<float>();
			foreach (var p in history.SegmentPacings)
				list.Add(p.GetOverallPacing());
			return list;
		}

		// ======================================================
		// DRAWING
		// ======================================================

		private void DrawGrid(Rect rect, int count)
		{
			Handles.BeginGUI();
			Handles.color = new Color(0.3f, 0.3f, 0.3f, 0.4f);

			float left = rect.x + padding;
			float right = rect.x + rect.width - padding;
			float top = rect.y + padding;
			float bottom = rect.y + rect.height - padding;

			float width = right - left;
			float height = bottom - top;

			int xSteps = Mathf.Max(1, count - 1);

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

		private void DrawCurve(Rect rect, List<float> list, Color color)
		{
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

			Vector2 GetPoint(int i)
			{
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

		private void DrawTargetOverlay(Rect rect, PacingTargetConfig target, int actualDataCount)
		{
			// The target config has a fixed number of segments (e.g., 25)
			// We need to resample it to match the actual data count for proper alignment
			var resampledThreat = ResampleCurve(target.ThreatTargets, actualDataCount);
			var resampledTempo = ResampleCurve(target.TempoTargets, actualDataCount);

			DrawDashed(rect, resampledThreat, Color.red);
			DrawDashed(rect, resampledTempo, Color.cyan);
		}

		private void DrawDashed(Rect rect, List<float> list, Color color)
		{
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

			Vector2 GetPoint(int i)
			{
				float x = left + i / (float)(list.Count - 1) * width;
				float y = bottom - Mathf.Clamp01(list[i]) * height;
				return new Vector2(x, y);
			}

			// Draw dotted lines between consecutive points
			for (int i = 0; i < list.Count - 1; i++)
				Handles.DrawDottedLine(GetPoint(i), GetPoint(i + 1), 4f);

			// Draw small circles at each point for better visibility
			for (int i = 0; i < list.Count; i++)
				Handles.DrawSolidDisc(GetPoint(i), Vector3.forward, pointRadius * 0.6f);

			Handles.EndGUI();
		}

		private void DrawLegend(Rect rect)
		{
			float boxWidth = 110f;
			float boxHeight = 60f;
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

		private void DrawLegendItem(Rect legendRect, int row, Color color, string label)
		{
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

		private void DrawBotsSection(PacingController controller)
		{
			showBots = EditorGUILayout.Foldout(showBots, "Bot Details", true);
			if (!showBots) return;

			EditorGUILayout.BeginVertical("box");
			EditorGUILayout.LabelField($"{botName}  |  Score: {botScore}");
			EditorGUILayout.LabelField($"{botName1}  |  Score: {botScore1}");

			EditorGUILayout.EndVertical();
		}

		private void DrawPacingDetails(GamePacing history)
		{
			showPacingDetails = EditorGUILayout.Foldout(showPacingDetails, "Pacing Details (Aspects / Factors)", true);
			if (!showPacingDetails) return;

			EditorGUILayout.BeginVertical("box");

			bool useScroll = history.SegmentPacings.Count > 10;

			if (useScroll)
				pacingScroll = EditorGUILayout.BeginScrollView(pacingScroll, GUILayout.Height(200));

			for (int i = 0; i < history.SegmentPacings.Count; i++)
			{
				var p = history.SegmentPacings[i];

				EditorGUILayout.LabelField($"Segment {i}", EditorStyles.boldLabel);

				EditorGUILayout.LabelField($"Threat: {p.Threat.Value:F3}");
				EditorGUILayout.LabelField($"Tempo: {p.Tempo.Value:F3}");
				// Modify the block code below because we now have GetFactorsInfo() in threat and tempo 
				//foreach (var factor in p.Threat. Factors) {
				//	EditorGUILayout.LabelField(
				//		$"   {factor.Key}: {factor.Value:F3}");
				//}
				foreach (var info in p.Threat.GetFactorsInfo())
				{
					// Now show the factor and its value. 
					EditorGUILayout.LabelField($"{info.factor}: {info.value:F3}");
					//EditorGUILayout.LabelField(
					//	$"   {info.Name}: {info.Value:F3}");
				}
				foreach (var info in p.Tempo.GetFactorsInfo())
				{
					// Now show the factor and its value. 
					EditorGUILayout.LabelField($"{info.factor}: {info.value:F3}");
					//EditorGUILayout.LabelField(
					//	$"   {info.Name}: {info.Value:F3}");
				}

				EditorGUILayout.Space(4);
			}

			if (useScroll)
				EditorGUILayout.EndScrollView();

			EditorGUILayout.EndVertical();
		}

		private void DrawSegmentDetails(GamePacing history)
		{
			showSegmentDetails = EditorGUILayout.Foldout(showSegmentDetails, "Segment Raw Data", true);
			if (!showSegmentDetails) return;

			EditorGUILayout.BeginVertical("box");

			bool useScroll = history.SegmentPacings.Count > 10;

			if (useScroll)
				segmentScroll = EditorGUILayout.BeginScrollView(segmentScroll, GUILayout.Height(200));

			for (int i = 0; i < history.SegmentPacings.Count; i++)
			{
				var info = history.SegmentGameplayDatas[i];
				EditorGUILayout.LabelField($"Segment {i} Counts", EditorStyles.boldLabel);
				EditorGUILayout.LabelField($"Collisions: {info.Collisions.Count}");
				EditorGUILayout.LabelField($"Angles: {info.Angles.Count}");
				EditorGUILayout.LabelField($"SafeDist: {info.SafeDistances.Count}");
				EditorGUILayout.LabelField($"Actions: {info.Actions.Count}");
				EditorGUILayout.LabelField($"BotDist: {info.BotsDistances.Count}");
				EditorGUILayout.LabelField($"Velocity: {info.Velocities.Count}");
				EditorGUILayout.Space(4);
			}

			if (useScroll)
				EditorGUILayout.EndScrollView();

			EditorGUILayout.EndVertical();
		}

		private void DrawSegmentEvaluation(
			List<float> actualThreat,
			List<float> actualTempo)
		{
			if (targetConfig == null)
				return;

			showEvaluationDetails = EditorGUILayout.Foldout(showEvaluationDetails, "Per Segment Evaluation", true);
			if (!showEvaluationDetails) return;

			EditorGUILayout.BeginVertical("box");

			var alignedThreat = ResampleCurve(targetConfig.ThreatTargets, actualThreat.Count);
			var alignedTempo = ResampleCurve(targetConfig.TempoTargets, actualTempo.Count);

			bool useScroll = actualThreat.Count > 10;

			if (useScroll)
				evaluationScroll = EditorGUILayout.BeginScrollView(evaluationScroll, GUILayout.Height(200));

			for (int i = 0; i < actualThreat.Count; i++)
			{
				float threatDiff = actualThreat[i] - alignedThreat[i];
				float tempoDiff = actualTempo[i] - alignedTempo[i];

				float mse = threatDiff * threatDiff + tempoDiff * tempoDiff;

				EditorGUILayout.LabelField(
					$"Segment {i}  |  ThreatΔ: {threatDiff:F3}  TempoΔ: {tempoDiff:F3}  Error: {mse:F4}");
			}

			if (useScroll)
				EditorGUILayout.EndScrollView();

			EditorGUILayout.EndVertical();
		}

		// ======================================================
		// EVALUATION
		// ======================================================

		private void DrawEvaluation(List<float> threat, List<float> tempo)
		{
			if (targetConfig == null) return;

			float threatError = CalculateMSE(threat, targetConfig.ThreatTargets);
			float tempoError = CalculateMSE(tempo, targetConfig.TempoTargets);
			float threatAvg = threat.Sum() / threat.Count;
			float tempoAvg = tempo.Sum() / tempo.Count;
			float overall = (threatAvg + tempoAvg) / 2;

			EditorGUILayout.Space();
			EditorGUILayout.LabelField(
				$"Threat AVG: {threatAvg:F2}, MSE: {threatError:F4}    Tempo AVG: {tempoAvg:F2}, MSE: {tempoError:F4}",
				EditorStyles.boldLabel);
			EditorGUILayout.LabelField(
			$"Overall Pacing: {overall:F2}",
			EditorStyles.boldLabel);
		}

		private float CalculateMSE(List<float> a, List<float> b)
		{
			if (a == null || b == null) return 0f;

			int count = Mathf.Min(a.Count, b.Count);
			if (count == 0) return 0f;

			float error = 0f;
			for (int i = 0; i < count; i++)
			{
				float d = a[i] - b[i];
				error += d * d;
			}

			return error / count;
		}

		private List<float> ResampleCurve(List<float> source, int targetCount)
		{
			var result = new List<float>();
			if (source == null || source.Count == 0 || targetCount <= 0)
				return result;

			for (int i = 0; i < targetCount; i++)
			{
				float t = i / (float)(targetCount - 1);
				float srcIndex = t * (source.Count - 1);

				int i0 = Mathf.FloorToInt(srcIndex);
				int i1 = Mathf.Min(i0 + 1, source.Count - 1);

				float lerp = srcIndex - i0;
				float value = Mathf.Lerp(source[i0], source[i1], lerp);

				result.Add(value);
			}

			return result;
		}
	}

	// [Todo] Add sections to show these later: 
	// 1. Show bots details: name, scores. 
	// 2. Show pacing details (use scroll area if data > 10): aspects and factors values. 
	// 3. Show segment details (use scroll area if data > 10): such as collisions<>, angles<>,safedist<>, actions<>, botdist<>, velocity<>.
	// 4. show per segment evaluation (use scroll area if data > 10). 


}