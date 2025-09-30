using UnityEngine;
using System.IO;

public static class Logger
{
#if UNITY_EDITOR || DEVELOPMENT_BUILD
    private static readonly bool EnableLogs = true;
#else
    private static readonly bool EnableLogs = false;
#endif

    public static string BatchLogPath;

    private static bool IsBatchMode => Application.isBatchMode;

    public static void Info(string message, bool writeToFile = false)
    {
        if (!EnableLogs && !IsBatchMode) return;
        string formatted = $"[INFO] {message}";

        Debug.Log(formatted);
        MaybeWriteToFile(formatted, writeToFile);
    }

    public static void Warning(string message, bool writeToFile = false)
    {
        if (!EnableLogs && !IsBatchMode) return;
        string formatted = $"[WARN] {message}";

        Debug.LogWarning(formatted);
        MaybeWriteToFile(formatted, writeToFile);
    }

    public static void Error(object message, bool writeToFile = false)
    {
        if (!EnableLogs && !IsBatchMode) return;
        string formatted = $"[ERROR] {message}";

        Debug.LogError(formatted);
        MaybeWriteToFile(formatted, writeToFile);
    }
    
    public static void Error(string message, bool writeToFile = false)
    {
        if (!EnableLogs && !IsBatchMode) return;
        string formatted = $"[ERROR] {message}";

        Debug.LogError(formatted);
        MaybeWriteToFile(formatted, writeToFile);
    }

    private static void MaybeWriteToFile(string message, bool writeToFile)
    {
        if (IsBatchMode && writeToFile)
        {
            try
            {
                File.AppendAllText(BatchLogPath,
                    $"{System.DateTime.Now:HH:mm:ss} {message}\n");
            }
            catch { /* ignore file errors */ }
        }
    }
}
