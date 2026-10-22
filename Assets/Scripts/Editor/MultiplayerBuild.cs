using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace SumoMultiplayer.Editor
{
    public static class MultiplayerBuild
    {
        private const string OutputPath = "Builds/Windows/Sumobot.exe";

        [MenuItem("Sumobot/Multiplayer/Build Windows Test Client")]
        public static void BuildWindowsTestClient()
        {
            string[] scenes = EditorBuildSettings.scenes
                .Where(scene => scene.enabled)
                .Select(scene => scene.path)
                .ToArray();

            if (scenes.Length == 0)
                throw new InvalidOperationException("No enabled scenes are configured in Build Settings.");

            string outputDirectory = Path.GetDirectoryName(OutputPath);
            if (!string.IsNullOrEmpty(outputDirectory))
                Directory.CreateDirectory(outputDirectory);

            bool oldSingleInstance = PlayerSettings.forceSingleInstance;
            bool oldRunInBackground = PlayerSettings.runInBackground;
            FullScreenMode oldFullscreenMode = PlayerSettings.fullScreenMode;
            int oldWidth = PlayerSettings.defaultScreenWidth;
            int oldHeight = PlayerSettings.defaultScreenHeight;

            try
            {
                PlayerSettings.forceSingleInstance = false;
                PlayerSettings.runInBackground = true;
                PlayerSettings.fullScreenMode = FullScreenMode.Windowed;
                PlayerSettings.defaultScreenWidth = 960;
                PlayerSettings.defaultScreenHeight = 540;

                var options = new BuildPlayerOptions
                {
                    scenes = scenes,
                    locationPathName = OutputPath,
                    target = BuildTarget.StandaloneWindows64,
                    options = BuildOptions.Development
                };

                BuildReport report = BuildPipeline.BuildPlayer(options);
                if (report.summary.result != BuildResult.Succeeded)
                {
                    throw new InvalidOperationException(
                        $"Multiplayer test build failed: {report.summary.result} " +
                        $"({report.summary.totalErrors} errors).");
                }

                Debug.Log(
                    $"[Online] Test client built at {Path.GetFullPath(OutputPath)} " +
                    $"({report.summary.totalSize} bytes).");
            }
            finally
            {
                PlayerSettings.forceSingleInstance = oldSingleInstance;
                PlayerSettings.runInBackground = oldRunInBackground;
                PlayerSettings.fullScreenMode = oldFullscreenMode;
                PlayerSettings.defaultScreenWidth = oldWidth;
                PlayerSettings.defaultScreenHeight = oldHeight;
                AssetDatabase.SaveAssets();
            }
        }
    }
}
