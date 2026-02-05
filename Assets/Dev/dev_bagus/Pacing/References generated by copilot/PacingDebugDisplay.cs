using UnityEngine;
using SumoBot;
using System.Linq;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace SumoBot.PacingDebug
{
    /// <summary>
    /// Custom inspector display for PacedAgent, showing pacing curves and action statistics.
    /// Can be attached to a GameObject in the scene to visualize pacing in real-time during play mode.
    /// </summary>
    public class PacingDebugDisplay : MonoBehaviour
    {
        public PacedAgent pacedAgent;
        public bool showPacingCurve = true;
        public bool showActionStatistics = true;
        public bool showPacingFactors = true;

        private void OnGUI()
        {
            if (pacedAgent == null || !showPacingCurve && !showActionStatistics && !showPacingFactors)
                return;

            GUILayout.BeginArea(new Rect(10, 10, 400, 600));
            GUILayout.BeginVertical(GUI.skin.box);

            GUILayout.Label($"Pacing Debug - {pacedAgent.ID}", EditorStyles.boldLabel);

            if (showPacingCurve)
            {
                DrawPacingCurveInfo();
            }

            if (showActionStatistics)
            {
                DrawActionStatistics();
            }

            if (showPacingFactors)
            {
                DrawCurrentPacingFactors();
            }

            GUILayout.EndVertical();
            GUILayout.EndArea();
        }

        private void DrawPacingCurveInfo()
        {
            GUILayout.Label("Pacing Curve", EditorStyles.boldLabel);

            var history = pacedAgent.GetPacingHistory();
            if (history.Count > 0)
            {
                var latestFrame = history[^1];
                GUILayout.Label($"Segment: {latestFrame.segmentIndex}", EditorStyles.miniLabel);
                GUILayout.Label($"Elapsed: {latestFrame.elapsed:F2}s", EditorStyles.miniLabel);
                GUILayout.Label($"Current Pacing: {latestFrame.overall:F3}", EditorStyles.miniLabel);
                GUILayout.Label($"Target Pacing: {latestFrame.target:F3}", EditorStyles.miniLabel);
                GUILayout.Label($"Threat: {latestFrame.threat:F3}", EditorStyles.miniLabel);
                GUILayout.Label($"Tempo: {latestFrame.tempo:F3}", EditorStyles.miniLabel);

                // Simple bar visualization
                DrawProgressBar(latestFrame.overall, "Current");
                DrawProgressBar(latestFrame.target, "Target");
            }

            GUILayout.Space(10);
        }

        private void DrawActionStatistics()
        {
            GUILayout.Label("Action Statistics", EditorStyles.boldLabel);

            var history = pacedAgent.GetActionHistory();
            GUILayout.Label($"Original Actions: {history.OriginalActionCount}", EditorStyles.miniLabel);
            GUILayout.Label($"Filtered Actions: {history.FilteredActionCount}", EditorStyles.miniLabel);
            GUILayout.Label($"Filtering Rate: {(history.FilteringRate * 100f):F1}%", EditorStyles.miniLabel);

            if (history.ActionTypeCount.Count > 0)
            {
                GUILayout.Label("Original by Type:", EditorStyles.miniLabel);
                foreach (var kvp in history.ActionTypeCount.OrderByDescending(x => x.Value))
                {
                    GUILayout.Label($"  {kvp.Key}: {kvp.Value}", EditorStyles.miniLabel);
                }
            }

            if (history.FilteredActionTypeCount.Count > 0)
            {
                GUILayout.Label("Filtered by Type:", EditorStyles.miniLabel);
                foreach (var kvp in history.FilteredActionTypeCount.OrderByDescending(x => x.Value))
                {
                    GUILayout.Label($"  {kvp.Key}: {kvp.Value}", EditorStyles.miniLabel);
                }
            }

            GUILayout.Space(10);
        }

        private void DrawCurrentPacingFactors()
        {
            GUILayout.Label("Pacing Factors", EditorStyles.boldLabel);

            var history = pacedAgent.GetPacingHistory();
            if (history.Count > 0)
            {
                var latestFrame = history[^1];
                var factors = latestFrame.factors;

                GUILayout.Label("Threat Factors:", EditorStyles.miniLabel);
                DrawFactor("  Collision", factors.collision);
                DrawFactor("  Enemy Skill", factors.enemySkill);
                DrawFactor("  Delta Angle", factors.deltaAngle);
                DrawFactor("  Delta Distance", factors.deltaDistance);

                GUILayout.Label("Tempo Factors:", EditorStyles.miniLabel);
                DrawFactor("  Action Intensity", factors.actionIntensity);
                DrawFactor("  Action Density", factors.actionDensity);
                DrawFactor("  Distance to Enemy", factors.avgDistanceToEnemy);
                DrawFactor("  Delta Velocity", factors.deltaVelocity);
            }
        }

        private void DrawFactor(string label, float value)
        {
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, GUILayout.Width(150));
            GUILayout.Label($"{value:F3}", GUILayout.Width(50));
            DrawProgressBar(value, "", 150);
            GUILayout.EndHorizontal();
        }

        private void DrawProgressBar(float value, string label, float width = 200)
        {
            GUILayout.BeginHorizontal(GUILayout.Width(width));
            if (!string.IsNullOrEmpty(label))
                GUILayout.Label(label, GUILayout.Width(60));

            value = Mathf.Clamp01(value);
            Rect barRect = EditorGUILayout.GetControlRect(GUILayout.Height(20), GUILayout.ExpandWidth(true));
            GUI.Box(barRect, "");
            Rect fillRect = new Rect(barRect.x, barRect.y, barRect.width * value, barRect.height);
            GUI.Box(fillRect, "", GUI.skin.GetStyle("Box"));
            GUI.Label(barRect, $"{value:P0}", new GUIStyle(GUI.skin.label) { alignment = TextAnchor.MiddleCenter });

            GUILayout.EndHorizontal();
        }
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(PacingDebugDisplay))]
    public class PacingDebugDisplayEditor : UnityEditor.Editor
    {
        public override void OnInspectorGUI()
        {
            DrawDefaultInspector();

            PacingDebugDisplay display = (PacingDebugDisplay)target;

            EditorGUILayout.Space();
            EditorGUILayout.LabelField("Debug Options", EditorStyles.boldLabel);

            if (display.pacedAgent != null)
            {
                var history = display.pacedAgent.GetPacingHistory();
                if (history.Count > 0)
                {
                    EditorGUILayout.LabelField("Pacing Curve", EditorStyles.boldLabel);

                    EditorGUILayout.LabelField($"Segments: {history.Count}");
                    EditorGUILayout.LabelField($"Avg Pacing: {history.Average(f => f.overall):F3}");
                    EditorGUILayout.LabelField($"Avg Target: {history.Average(f => f.target):F3}");

                    // Draw a mini curve preview
                    EditorGUILayout.LabelField("Pacing Over Time", EditorStyles.boldLabel);
                    DrawCurvePreview(history);
                }

                var actionHistory = display.pacedAgent.GetActionHistory();
                EditorGUILayout.Space();
                EditorGUILayout.LabelField("Action History", EditorStyles.boldLabel);
                EditorGUILayout.LabelField($"Total Actions: {actionHistory.OriginalActionCount}");
                EditorGUILayout.LabelField($"Filtered Actions: {actionHistory.FilteredActionCount}");
                EditorGUILayout.LabelField($"Filtering Rate: {(actionHistory.FilteringRate * 100f):F1}%");

                if (actionHistory.FilteredActionTypeCount.Count > 0)
                {
                    EditorGUILayout.LabelField("Filtered Breakdown:", EditorStyles.boldLabel);
                    foreach (var kvp in actionHistory.FilteredActionTypeCount)
                    {
                        EditorGUILayout.LabelField($"  {kvp.Key}", $"{kvp.Value}");
                    }
                }
            }
        }

        private void DrawCurvePreview(System.Collections.Generic.List<PacingFrame> history)
        {
            if (history.Count == 0)
                return;

            Rect rect = GUILayoutUtility.GetRect(300, 100);
            GUI.Box(rect, "");

            float minPacing = 0f;
            float maxPacing = 1f;
            float width = rect.width - 10;
            float height = rect.height - 10;

            for (int i = 0; i < history.Count - 1; i++)
            {
                float x1 = rect.x + 5 + (i / (float)(history.Count - 1)) * width;
                float y1 = rect.y + rect.height - 5 - (history[i].overall - minPacing) / (maxPacing - minPacing) * height;

                float x2 = rect.x + 5 + ((i + 1) / (float)(history.Count - 1)) * width;
                float y2 = rect.y + rect.height - 5 - (history[i + 1].overall - minPacing) / (maxPacing - minPacing) * height;

                Handles.color = Color.cyan;
                Handles.DrawLine(new Vector3(x1, y1, 0), new Vector3(x2, y2, 0));

                // Draw target line
                float ty1 = rect.y + rect.height - 5 - (history[i].target - minPacing) / (maxPacing - minPacing) * height;
                float ty2 = rect.y + rect.height - 5 - (history[i + 1].target - minPacing) / (maxPacing - minPacing) * height;

                Handles.color = Color.yellow;
                Handles.DrawLine(new Vector3(x1, ty1, 0), new Vector3(x2, ty2, 0));
            }

            // Draw legend
            EditorGUI.LabelField(new Rect(rect.x + 5, rect.y + 5, 100, 20), "Cyan: Actual", EditorStyles.miniLabel);
            EditorGUI.LabelField(new Rect(rect.x + 150, rect.y + 5, 100, 20), "Yellow: Target", EditorStyles.miniLabel);
        }
    }
#endif
}
