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

        [MenuItem("Tools/Open Simulation Setting")]
        public static void OpenSimulationSettingFolder()
        {
            string folder = Path.Combine(Application.persistentDataPath, "Settings");

            if (!Directory.Exists(folder))
                Directory.CreateDirectory(folder);

            EditorUtility.RevealInFinder(folder);
        }

        [MenuItem("Tools/Open Simulation Result")]
        public static void OpenResultSimulationFolder()
        {
            string folder = Path.Combine(Application.persistentDataPath, "Simulation");

            if (!Directory.Exists(folder))
                Directory.CreateDirectory(folder);

            EditorUtility.RevealInFinder(folder);
        }
    }
}
#endif
