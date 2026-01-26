#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoHelper;

[CustomPropertyDrawer(typeof(SimulationSetting))]
public class SimulationSettingDrawer : PropertyDrawer
{
    private List<System.Type> botTypes;
    private string[] botIDs;
    private Dictionary<string, bool> selectedAgents = new Dictionary<string, bool>();
    private Dictionary<int, bool> selectedTimers = new Dictionary<int, bool>();
    private Dictionary<float, bool> selectedActionIntervals = new Dictionary<float, bool>();
    private Dictionary<SumoManager.RoundSystem, bool> selectedRoundSystems = new Dictionary<SumoManager.RoundSystem, bool>();
    private Dictionary<SumoCore.SkillType, bool> selectedSkills = new Dictionary<SumoCore.SkillType, bool>();

    private bool showBotSelection = true;
    private bool showTimerSelection = true;
    private bool showActionIntervalSelection = true;
    private bool showRoundSystemSelection = true;
    private bool showSkillSelection = true;
    private bool initialized = false;

    private void Initialize(SerializedProperty property)
    {
        if (initialized) return;

        botTypes = BotUtility.GetAllBotTypes();

        // Get bot IDs by creating instances
        botIDs = new string[botTypes.Count];
        for (int i = 0; i < botTypes.Count; i++)
        {
            var instance = ScriptableObject.CreateInstance(botTypes[i]) as Bot;
            if (instance != null)
            {
                botIDs[i] = instance.ID;
                ScriptableObject.DestroyImmediate(instance);
            }
        }

        LoadSettings(property);
        initialized = true;
    }

    private void LoadSettings(SerializedProperty property)
    {
        var selectedAgentsProp = property.FindPropertyRelative("SelectedAgents");
        var timersProp = property.FindPropertyRelative("Timers");
        var intervalsProp = property.FindPropertyRelative("ActionIntervals");
        var roundsProp = property.FindPropertyRelative("RoundSystem");
        var skillsProp = property.FindPropertyRelative("Skills");

        // Load agent selections
        selectedAgents.Clear();
        for (int i = 0; i < botIDs.Length; i++)
        {
            string botID = botIDs[i];
            bool isSelected = false;

            if (selectedAgentsProp != null && selectedAgentsProp.arraySize > 0)
            {
                for (int j = 0; j < selectedAgentsProp.arraySize; j++)
                {
                    if (selectedAgentsProp.GetArrayElementAtIndex(j).stringValue == botID)
                    {
                        isSelected = true;
                        break;
                    }
                }
            }
            else
            {
                isSelected = true; // Default to all selected
            }

            selectedAgents[botID] = isSelected;
        }

        // Load timer selections
        selectedTimers.Clear();
        int[] defaultTimers = new int[] { 15, 30, 45, 60 };
        foreach (int timer in defaultTimers)
        {
            bool isSelected = false;
            if (timersProp != null && timersProp.arraySize > 0)
            {
                for (int i = 0; i < timersProp.arraySize; i++)
                {
                    if (timersProp.GetArrayElementAtIndex(i).intValue == timer)
                    {
                        isSelected = true;
                        break;
                    }
                }
            }
            selectedTimers[timer] = isSelected;
        }

        // Load action interval selections
        selectedActionIntervals.Clear();
        float[] defaultIntervals = new float[] { 0.1f, 0.2f, 0.5f };
        foreach (float interval in defaultIntervals)
        {
            bool isSelected = false;
            if (intervalsProp != null && intervalsProp.arraySize > 0)
            {
                for (int i = 0; i < intervalsProp.arraySize; i++)
                {
                    if (Mathf.Approximately(intervalsProp.GetArrayElementAtIndex(i).floatValue, interval))
                    {
                        isSelected = true;
                        break;
                    }
                }
            }
            selectedActionIntervals[interval] = isSelected;
        }

        // Load round system selections
        selectedRoundSystems.Clear();
        SumoManager.RoundSystem[] defaultRounds = new SumoManager.RoundSystem[] {
            SumoManager.RoundSystem.BestOf1,
            SumoManager.RoundSystem.BestOf3,
            SumoManager.RoundSystem.BestOf5
        };
        foreach (var round in defaultRounds)
        {
            bool isSelected = false;
            if (roundsProp != null && roundsProp.arraySize > 0)
            {
                for (int i = 0; i < roundsProp.arraySize; i++)
                {
                    var element = roundsProp.GetArrayElementAtIndex(i);
                    var enumValue = (SumoManager.RoundSystem)element.intValue;
                    if (enumValue == round)
                    {
                        isSelected = true;
                        break;
                    }
                }
            }
            selectedRoundSystems[round] = isSelected;
        }

        // Load skill selections
        selectedSkills.Clear();
        SumoCore.SkillType[] defaultSkills = new SumoCore.SkillType[] {
            SumoCore.SkillType.Boost,
            SumoCore.SkillType.Stone
        };
        foreach (var skill in defaultSkills)
        {
            bool isSelected = false;
            if (skillsProp != null && skillsProp.arraySize > 0)
            {
                for (int i = 0; i < skillsProp.arraySize; i++)
                {
                    var element = skillsProp.GetArrayElementAtIndex(i);
                    var enumValue = (SumoCore.SkillType)element.intValue;
                    if (enumValue == skill)
                    {
                        isSelected = true;
                        break;
                    }
                }
            }
            selectedSkills[skill] = isSelected;
        }
    }

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        return EditorGUIUtility.singleLineHeight; // We'll use OnGUI with auto-layout instead
    }

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        Initialize(property);

        EditorGUI.BeginProperty(position, label, property);

        // Draw foldout
        property.isExpanded = EditorGUI.Foldout(position, property.isExpanded, label, true);

        if (property.isExpanded)
        {
            EditorGUI.indentLevel++;

            // Draw Iteration field
            var iterationProp = property.FindPropertyRelative("Iteration");
            EditorGUILayout.PropertyField(iterationProp);

            EditorGUILayout.Space(10);

            // Bot Selection
            DrawBotSelection(property);
            EditorGUILayout.Space(5);

            // Timer Selection
            DrawTimerSelection(property);
            EditorGUILayout.Space(5);

            // Action Interval Selection
            DrawActionIntervalSelection(property);
            EditorGUILayout.Space(5);

            // Round System Selection
            DrawRoundSystemSelection(property);
            EditorGUILayout.Space(5);

            // Skill Selection
            DrawSkillSelection(property);

            EditorGUI.indentLevel--;
        }

        EditorGUI.EndProperty();
    }

    private void DrawBotSelection(SerializedProperty property)
    {
        showBotSelection = EditorGUILayout.Foldout(showBotSelection, "Bot Selection", true, EditorStyles.foldoutHeader);
        if (showBotSelection)
        {
            EditorGUI.indentLevel++;

            bool anyChanged = false;
            for (int i = 0; i < botIDs.Length; i++)
            {
                string botID = botIDs[i];
                bool oldValue = selectedAgents.ContainsKey(botID) && selectedAgents[botID];
                bool newValue = EditorGUILayout.Toggle(botID, oldValue);

                if (oldValue != newValue)
                {
                    selectedAgents[botID] = newValue;
                    anyChanged = true;
                }
            }

            if (anyChanged)
            {
                var selectedAgentsProp = property.FindPropertyRelative("SelectedAgents");
                var selected = selectedAgents.Where(kvp => kvp.Value).Select(kvp => kvp.Key).ToArray();

                selectedAgentsProp.arraySize = selected.Length;
                for (int i = 0; i < selected.Length; i++)
                {
                    selectedAgentsProp.GetArrayElementAtIndex(i).stringValue = selected[i];
                }
                property.serializedObject.ApplyModifiedProperties();
            }

            int selectedCount = selectedAgents.Values.Count(v => v);
            if (selectedCount < 2)
                EditorGUILayout.HelpBox($"Please select at least 2 agents to run the simulation. Currently selected: {selectedCount}", MessageType.Warning);

            EditorGUI.indentLevel--;
        }
    }

    private void DrawTimerSelection(SerializedProperty property)
    {
        showTimerSelection = EditorGUILayout.Foldout(showTimerSelection, "Timer Selection", true, EditorStyles.foldoutHeader);
        if (showTimerSelection)
        {
            EditorGUI.indentLevel++;

            bool anyChanged = false;
            int[] defaultTimers = new int[] { 15, 30, 45, 60 };
            foreach (int timer in defaultTimers)
            {
                bool oldValue = selectedTimers.ContainsKey(timer) && selectedTimers[timer];
                bool newValue = EditorGUILayout.Toggle($"{timer} seconds", oldValue);

                if (oldValue != newValue)
                {
                    selectedTimers[timer] = newValue;
                    anyChanged = true;
                }
            }

            if (anyChanged)
            {
                var timersProp = property.FindPropertyRelative("Timers");
                var selected = selectedTimers.Where(kvp => kvp.Value).Select(kvp => kvp.Key).OrderBy(t => t).ToArray();

                timersProp.arraySize = selected.Length;
                for (int i = 0; i < selected.Length; i++)
                {
                    timersProp.GetArrayElementAtIndex(i).intValue = selected[i];
                }
                property.serializedObject.ApplyModifiedProperties();
            }

            int selectedTimerCount = selectedTimers.Values.Count(v => v);
            if (selectedTimerCount == 0)
                EditorGUILayout.HelpBox("Please select at least 1 timer value.", MessageType.Warning);

            EditorGUI.indentLevel--;
        }
    }

    private void DrawActionIntervalSelection(SerializedProperty property)
    {
        showActionIntervalSelection = EditorGUILayout.Foldout(showActionIntervalSelection, "Action Interval Selection", true, EditorStyles.foldoutHeader);
        if (showActionIntervalSelection)
        {
            EditorGUI.indentLevel++;

            bool anyChanged = false;
            float[] defaultIntervals = new float[] { 0.1f, 0.2f, 0.5f };
            foreach (float interval in defaultIntervals)
            {
                bool oldValue = selectedActionIntervals.ContainsKey(interval) && selectedActionIntervals[interval];
                bool newValue = EditorGUILayout.Toggle($"{interval} seconds", oldValue);

                if (oldValue != newValue)
                {
                    selectedActionIntervals[interval] = newValue;
                    anyChanged = true;
                }
            }

            if (anyChanged)
            {
                var intervalsProp = property.FindPropertyRelative("ActionIntervals");
                var selected = selectedActionIntervals.Where(kvp => kvp.Value).Select(kvp => kvp.Key).OrderBy(i => i).ToArray();

                intervalsProp.arraySize = selected.Length;
                for (int i = 0; i < selected.Length; i++)
                {
                    intervalsProp.GetArrayElementAtIndex(i).floatValue = selected[i];
                }
                property.serializedObject.ApplyModifiedProperties();
            }

            int selectedIntervalCount = selectedActionIntervals.Values.Count(v => v);
            if (selectedIntervalCount == 0)
                EditorGUILayout.HelpBox("Please select at least 1 action interval value.", MessageType.Warning);

            EditorGUI.indentLevel--;
        }
    }

    private void DrawRoundSystemSelection(SerializedProperty property)
    {
        showRoundSystemSelection = EditorGUILayout.Foldout(showRoundSystemSelection, "Round System Selection", true, EditorStyles.foldoutHeader);
        if (showRoundSystemSelection)
        {
            EditorGUI.indentLevel++;

            bool anyChanged = false;
            SumoManager.RoundSystem[] defaultRounds = new SumoManager.RoundSystem[] {
                SumoManager.RoundSystem.BestOf1,
                SumoManager.RoundSystem.BestOf3,
                SumoManager.RoundSystem.BestOf5
            };
            foreach (var round in defaultRounds)
            {
                bool oldValue = selectedRoundSystems.ContainsKey(round) && selectedRoundSystems[round];
                string roundLabel = round == SumoManager.RoundSystem.BestOf1 ? "1 Round" :
                                   round == SumoManager.RoundSystem.BestOf3 ? "3 Rounds" : "5 Rounds";
                bool newValue = EditorGUILayout.Toggle(roundLabel, oldValue);

                if (oldValue != newValue)
                {
                    selectedRoundSystems[round] = newValue;
                    anyChanged = true;
                }
            }

            if (anyChanged)
            {
                var roundsProp = property.FindPropertyRelative("RoundSystem");
                var selected = selectedRoundSystems.Where(kvp => kvp.Value).Select(kvp => kvp.Key).OrderBy(r => r).ToArray();

                roundsProp.arraySize = selected.Length;
                for (int i = 0; i < selected.Length; i++)
                {
                    roundsProp.GetArrayElementAtIndex(i).intValue = (int)selected[i];
                }
                property.serializedObject.ApplyModifiedProperties();
            }

            int selectedRoundCount = selectedRoundSystems.Values.Count(v => v);
            if (selectedRoundCount == 0)
                EditorGUILayout.HelpBox("Please select at least 1 round system.", MessageType.Warning);

            EditorGUI.indentLevel--;
        }
    }

    private void DrawSkillSelection(SerializedProperty property)
    {
        showSkillSelection = EditorGUILayout.Foldout(showSkillSelection, "Skill Selection", true, EditorStyles.foldoutHeader);
        if (showSkillSelection)
        {
            EditorGUI.indentLevel++;

            bool anyChanged = false;
            SumoCore.SkillType[] defaultSkills = new SumoCore.SkillType[] {
                SumoCore.SkillType.Boost,
                SumoCore.SkillType.Stone
            };
            foreach (var skill in defaultSkills)
            {
                bool oldValue = selectedSkills.ContainsKey(skill) && selectedSkills[skill];
                bool newValue = EditorGUILayout.Toggle($"{skill}", oldValue);

                if (oldValue != newValue)
                {
                    selectedSkills[skill] = newValue;
                    anyChanged = true;
                }
            }

            if (anyChanged)
            {
                var skillsProp = property.FindPropertyRelative("Skills");
                var selected = selectedSkills.Where(kvp => kvp.Value).Select(kvp => kvp.Key).OrderBy(s => s).ToArray();

                skillsProp.arraySize = selected.Length;
                for (int i = 0; i < selected.Length; i++)
                {
                    skillsProp.GetArrayElementAtIndex(i).intValue = (int)selected[i];
                }
                property.serializedObject.ApplyModifiedProperties();
            }

            int selectedSkillCount = selectedSkills.Values.Count(v => v);
            if (selectedSkillCount == 0)
                EditorGUILayout.HelpBox("No skills selected. Default skills from bot scripts will be used.", MessageType.Info);

            EditorGUI.indentLevel--;
        }
    }
}
#endif
