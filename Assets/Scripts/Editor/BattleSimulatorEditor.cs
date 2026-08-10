#if UNITY_EDITOR
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;
using SumoHelper;

[CustomEditor(typeof(BattleSimulator))]
public class BattleSimulatorEditor : Editor
{
    // Foldout states
    private bool showSimpleSettings = true;
    private bool showAdvancedSettings = true;
    private bool showFocusBotSelection = true;


    public override void OnInspectorGUI()
    {
        BattleSimulator simulator = (BattleSimulator)target;
        if (simulator == null) return;

        serializedObject.Update();

        // Mode Selection
        EditorGUILayout.Space(5);
        EditorGUILayout.LabelField("Simulator Mode", EditorStyles.boldLabel);
        simulator.Mode = (SimulatorMode)EditorGUILayout.EnumPopup("Mode", simulator.Mode);

        EditorGUILayout.Space(10);

        // Simple Mode Settings
        if (simulator.Mode == SimulatorMode.Simple)
        {
            showSimpleSettings = EditorGUILayout.Foldout(showSimpleSettings, "Simple Mode Settings (Single)", true, EditorStyles.foldoutHeader);
            if (showSimpleSettings)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.HelpBox("Simple mode runs multiple battles with basic timescale control.", MessageType.Info);

                simulator.TotalSimulations = EditorGUILayout.IntField("Total Simulations", simulator.TotalSimulations);
                simulator.SimpleTimeScale = EditorGUILayout.FloatField("Time Scale", simulator.SimpleTimeScale);
                simulator.SwapAIInterval = EditorGUILayout.IntField("Swap AI Interval", simulator.SwapAIInterval);
                simulator.SimulationOnStart = EditorGUILayout.Toggle("Simulation On Start", simulator.SimulationOnStart);
                simulator.QuitAfterDone = EditorGUILayout.Toggle("Quit After Done", simulator.QuitAfterDone);

                EditorGUI.indentLevel--;
            }
        }
        // Advanced Mode Settings
        else if (simulator.Mode == SimulatorMode.Advanced)
        {
            showAdvancedSettings = EditorGUILayout.Foldout(showAdvancedSettings, "Advanced Mode Settings (Batch)", true, EditorStyles.foldoutHeader);
            if (showAdvancedSettings)
            {
                EditorGUI.indentLevel++;
                EditorGUILayout.HelpBox("Advanced mode runs batch simulations with various configurations.", MessageType.Info);

                simulator.IgnoreResume = EditorGUILayout.Toggle("Ignore Resume", simulator.IgnoreResume);
                simulator.ExcludeHeavyMLBots = EditorGUILayout.Toggle("Exclude Heavy Bots", simulator.ExcludeHeavyMLBots);
                simulator.DefaultTimeScale = EditorGUILayout.FloatField("Default Time Scale", simulator.DefaultTimeScale);
                simulator.SimulationOnStart = EditorGUILayout.Toggle("Simulation On Start", simulator.SimulationOnStart);
                simulator.RoundCountdown = EditorGUILayout.IntField("Round Countdown", simulator.RoundCountdown);

                EditorGUILayout.Space(10);
                EditorGUILayout.PropertyField(serializedObject.FindProperty("Setting"), new GUIContent("Simulation Setting"), true);

                EditorGUILayout.Space(10);
                EditorGUILayout.LabelField("Pacing Simulation", EditorStyles.boldLabel);
                simulator.PacingSimulation = EditorGUILayout.Toggle("Pacing Simulation", simulator.PacingSimulation);

                if (simulator.PacingSimulation)
                {
                    EditorGUILayout.HelpBox(
                        "Generates Focus-vs-Rest matchups (both sides mirrored) between the bots checked below and the rest of " +
                        "Simulation Setting > Selected Agents, swept across every file in Sim Targets Folder x Sim Constraints Folder. " +
                        "Only the focus bot's side has pacing (action filtering) applied; the rest bot plays unmodified. Replaces the " +
                        "default full round-robin matchup generation.",
                        MessageType.Info);

                    EditorGUI.indentLevel++;
                    simulator.SimTargetsFolder = EditorGUILayout.TextField("Pacing Targets Folder", simulator.SimTargetsFolder);
                    simulator.SimConstraintsFolder = EditorGUILayout.TextField("Pacing Constraints Folder", simulator.SimConstraintsFolder);
                    simulator.PacingSegmentDuration = EditorGUILayout.IntField("Pacing Segment Duration", simulator.PacingSegmentDuration);
                    simulator.PacingCollisionWindow = EditorGUILayout.IntField("Pacing Collision Window", simulator.PacingCollisionWindow);
                    simulator.PacingMax = EditorGUILayout.FloatField("Pacing Max", simulator.PacingMax);
                    simulator.PacingMin = EditorGUILayout.FloatField("Pacing Min", simulator.PacingMin);

                    EditorGUILayout.Space(5);
                    simulator.IncludeFocusMatchups = EditorGUILayout.Toggle(
                        new GUIContent("Include Focus Matchups", "Also sweep Focus bots against each other, each one taking a turn as the pacing-filtered side. Requires at least 2 Focus bots."),
                        simulator.IncludeFocusMatchups);

                    EditorGUILayout.Space(5);
                    DrawFocusBotSelection(simulator);
                    EditorGUI.indentLevel--;
                }

                EditorGUI.indentLevel--;
            }
        }

        serializedObject.ApplyModifiedProperties();

        if (GUI.changed)
        {
            EditorUtility.SetDirty(simulator);
        }
    }

    /// <summary>
    /// One checkbox per bot currently in Setting.SelectedAgents, letting the user explicitly
    /// mark which ones are "focus" bots for Pacing Simulation matchup generation. Checkbox order
    /// (not the underlying array order, which follows bot registration order and isn't user
    /// controllable) is what determines focus vs rest.
    /// </summary>
    private void DrawFocusBotSelection(BattleSimulator simulator)
    {
        showFocusBotSelection = EditorGUILayout.Foldout(showFocusBotSelection, "Focus Bot Selection", true, EditorStyles.foldoutHeader);
        if (!showFocusBotSelection) return;

        EditorGUI.indentLevel++;

        var selectedAgents = simulator.Setting?.SelectedAgents;
        if (selectedAgents == null || selectedAgents.Length == 0)
        {
            EditorGUILayout.HelpBox("Select agents in Simulation Setting > Bot Selection first.", MessageType.Warning);
            EditorGUI.indentLevel--;
            return;
        }

        var focusSet = new HashSet<string>(simulator.FocusBotIDs ?? new string[0]);
        bool anyChanged = false;

        foreach (var botID in selectedAgents)
        {
            bool oldValue = focusSet.Contains(botID);
            bool newValue = EditorGUILayout.Toggle(botID, oldValue);

            if (newValue && !oldValue) { focusSet.Add(botID); anyChanged = true; }
            else if (!newValue && oldValue) { focusSet.Remove(botID); anyChanged = true; }
        }

        if (anyChanged)
        {
            simulator.FocusBotIDs = focusSet.ToArray();
        }

        int focusCount = focusSet.Count;
        int restCount = selectedAgents.Length - focusCount;
        if (focusCount == 0 || restCount == 0)
            EditorGUILayout.HelpBox($"Need at least 1 focus bot and 1 rest bot. Currently: Focus={focusCount}, Rest={restCount}.", MessageType.Warning);
        if (simulator.IncludeFocusMatchups && focusCount < 2)
            EditorGUILayout.HelpBox($"Include Focus Matchups is enabled but needs at least 2 focus bots. Currently: Focus={focusCount}.", MessageType.Warning);

        EditorGUI.indentLevel--;
    }
}
#endif
