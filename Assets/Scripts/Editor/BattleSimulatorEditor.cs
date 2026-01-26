#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using SumoHelper;

[CustomEditor(typeof(BattleSimulator))]
public class BattleSimulatorEditor : Editor
{
    // Foldout states
    private bool showSimpleSettings = true;
    private bool showAdvancedSettings = true;


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

                simulator.DefaultTimeScale = EditorGUILayout.FloatField("Default Time Scale", simulator.DefaultTimeScale);
                simulator.SimulationOnStart = EditorGUILayout.Toggle("Simulation On Start", simulator.SimulationOnStart);
                simulator.RoundCountdown = EditorGUILayout.IntField("Round Countdown", simulator.RoundCountdown);

                EditorGUILayout.Space(10);
                EditorGUILayout.PropertyField(serializedObject.FindProperty("Setting"), new GUIContent("Simulation Setting"), true);

                EditorGUI.indentLevel--;
            }
        }

        serializedObject.ApplyModifiedProperties();

        if (GUI.changed)
        {
            EditorUtility.SetDirty(simulator);
        }
    }
}
#endif
