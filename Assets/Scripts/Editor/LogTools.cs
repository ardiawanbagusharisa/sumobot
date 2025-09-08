#if UNITY_EDITOR
namespace SumoEditor
{
    using System.IO;

    using UnityEditor;
    using UnityEngine;

    public static class Utility
    {
        [MenuItem("Tools/Open Log Folder")]
        public static void OpenLogFolder()
        {
            string logFolder = Path.Combine(Application.persistentDataPath, "Logs");

            if (!Directory.Exists(logFolder))
                Directory.CreateDirectory(logFolder);

        EditorUtility.RevealInFinder(logFolder);
            Debug.Log("Logs folder: " + logFolder);
        }
    }
}
#endif
