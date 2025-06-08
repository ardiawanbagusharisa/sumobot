#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;
using System.IO;
using CoreSumo;

public static class Utility
{
    [MenuItem("Tools/Open Log Folder")]
    public static void OpenLogFolder()
    {
        string logFolder = Path.Combine(Application.persistentDataPath, "Logs");

        if (!Directory.Exists(logFolder))
            Directory.CreateDirectory(logFolder);

#if UNITY_EDITOR
        EditorUtility.RevealInFinder(logFolder);
#else
        Debug.Log("Logs folder: " + logFolder);
#endif
    }

    public static LogActorType ToLogActorType(this PlayerSide side)
    {
        return side == PlayerSide.Left ? LogActorType.LeftPlayer : LogActorType.RightPlayer;
    }
}
