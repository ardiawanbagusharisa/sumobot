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
            string folder = Path.Combine(Application.persistentDataPath, "Logs");

            if (!Directory.Exists(folder))
                Directory.CreateDirectory(folder);

            EditorUtility.RevealInFinder(folder);
        }
    }
}
#endif
