namespace SumoEditor
{
    using System.IO;

#if UNITY_EDITOR
    using UnityEditor;
    using UnityEngine;
#endif

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
    }
}
