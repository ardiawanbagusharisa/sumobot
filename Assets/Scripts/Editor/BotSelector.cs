#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using SumoBot;
using SumoCore;

[CustomEditor(typeof(BotManager))]
public class BotSelector : Editor
{
    private List<Bot> botTypes;
    private string[] botNames;

    private void OnEnable()
    {
        botTypes = BotUtility.GetAllBotInstances();
        botNames = botTypes.ConvertAll(t => t.ID).ToArray();
    }

    public override void OnInspectorGUI()
    {
        BotManager selector = (BotManager)target;

        if (selector == null) return;

        DrawDefaultInspector();

        if (!selector.BotEnabled) return;


        EditorGUILayout.LabelField("Select Bots for Match", EditorStyles.boldLabel);

        if (Application.isPlaying)
        {
            GUI.enabled = false;
            
            selector.Left = (Bot)EditorGUILayout.ObjectField("Left (active)", selector.Left, typeof(Bot), false);
            selector.Right = (Bot)EditorGUILayout.ObjectField("Right (active)", selector.Right, typeof(Bot), false);

            GUI.enabled = true;
        }
        selector.leftBotIndex = EditorGUILayout.Popup("Left Bot", selector.leftBotIndex, botNames);
        selector.rightBotIndex = EditorGUILayout.Popup("Right Bot", selector.rightBotIndex, botNames);

        if (Application.isPlaying)
            if (GUILayout.Button("Assign"))
            {
                Bot left = botTypes[selector.leftBotIndex];
                Bot right = botTypes[selector.rightBotIndex];
                selector.Assign(left, PlayerSide.Left);
                selector.Assign(right, PlayerSide.Right);
                Logger.Info($"Assigned {botNames[selector.leftBotIndex]} (Left), {botNames[selector.rightBotIndex]} (Right)");
            }

        if (GUI.changed)
        {
            EditorUtility.SetDirty(selector);
        }
    }
}
#endif
