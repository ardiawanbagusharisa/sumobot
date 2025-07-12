#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using SumoBot;
using Codice.Client.BaseCommands;

[CustomEditor(typeof(BotManager))]
public class BotSelector : Editor
{
    private List<Type> botTypes;
    private string[] botNames;

    private void OnEnable()
    {
        botTypes = BotUtility.GetAllBotTypes();
        botNames = botTypes.ConvertAll(t => t.Name).ToArray();
    }

    public override void OnInspectorGUI()
    {
        BotManager selector = (BotManager)target;

        if (selector == null) return;
        DrawDefaultInspector();

        if (!selector.IsEnable || !selector.IsScriptable) return;

        GUI.enabled = Application.isPlaying;

        EditorGUILayout.LabelField("Select Bots for Match", EditorStyles.boldLabel);

        selector.leftBotIndex = EditorGUILayout.Popup("Left Bot", selector.leftBotIndex, botNames);
        selector.rightBotIndex = EditorGUILayout.Popup("Right Bot", selector.rightBotIndex, botNames);

        if (GUILayout.Button("Assign"))
        {
            selector.Left = CreateInstance(botTypes[selector.leftBotIndex]) as Bot;
            selector.Right = CreateInstance(botTypes[selector.rightBotIndex]) as Bot;
            Debug.Log($"Assigned {botNames[selector.leftBotIndex]} (Left), {botNames[selector.rightBotIndex]} (Right)");
        }

        if (GUI.changed)
        {
            EditorUtility.SetDirty(selector);
        }
    }
}
#endif
