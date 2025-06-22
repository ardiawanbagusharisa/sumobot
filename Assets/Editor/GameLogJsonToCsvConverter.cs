#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using Newtonsoft.Json.Linq;
using System.Text.RegularExpressions;

public class GameLogJsonToCsvConverter : EditorWindow
{
    private string inputFolder = "";
    private string outputFile = "";

    [MenuItem("Tools/Game Log JSON to CSV")]
    public static void ShowWindow()
    {
        GetWindow<GameLogJsonToCsvConverter>("JSON to CSV Converter");
    }

    private void OnGUI()
    {
        GUILayout.Label("Game Log JSON to CSV", EditorStyles.boldLabel);

        GUILayout.Space(10);
        GUILayout.Label("1. Select Input Folder (JSON files):");
        GUILayout.BeginHorizontal();
        inputFolder = EditorGUILayout.TextField(inputFolder);
        string logFolder = Path.Combine(Application.persistentDataPath, "Logs");
        if (GUILayout.Button("Browse", GUILayout.Width(80)))
        {
            inputFolder = EditorUtility.OpenFolderPanel("Select Input Folder", logFolder, "");
        }
        GUILayout.EndHorizontal();

        GUILayout.Space(10);
        GUILayout.Label("2. Select Output CSV File:");
        GUILayout.BeginHorizontal();
        outputFile = EditorGUILayout.TextField(outputFile);
        if (GUILayout.Button("Save As", GUILayout.Width(80)))
        {
            outputFile = EditorUtility.SaveFilePanel("Save CSV File", "", "game_logs.csv", "csv");
        }
        GUILayout.EndHorizontal();

        GUILayout.Space(20);
        if (GUILayout.Button("Convert to CSV", GUILayout.Height(40)))
        {
            if (string.IsNullOrEmpty(inputFolder) || string.IsNullOrEmpty(outputFile))
            {
                EditorUtility.DisplayDialog("Error", "Please select both input folder and output file.", "OK");
                return;
            }

            ConvertLogsToCsv(inputFolder, outputFile);
        }
    }

    private static int ExtractGameIndex(string filename)
    {
        // e.g., game_001 -> 1
        var match = Regex.Match(filename, @"game_(\d+)");
        return match.Success ? int.Parse(match.Groups[1].Value) : -1;
    }


    private void ConvertLogsToCsv(string folderPath, string outputPath)
    {
        try
        {
            var csvRows = new List<Dictionary<string, string>>();

            var files = Directory.GetFiles(folderPath, "game_*.json")
            .Where(f => Regex.IsMatch(Path.GetFileName(f), @"game_\d+\.json"))
            .OrderBy(f => ExtractGameIndex(Path.GetFileNameWithoutExtension(f)))
            .ToList();

            foreach (var file in files)
            {
                string json = File.ReadAllText(file);
                JObject root = JObject.Parse(json);

                int gameIndex = (int?)root["index"] ?? -1;
                string gameTimestamp = (string)root["timestamp"] ?? "";
                string gameWinner = root?["winner"]?.ToString() ?? "";

                foreach (var round in root?["rounds"] as JArray ?? new JArray())
                {
                    int roundIndex = (int?)round["index"] ?? -1;
                    string roundTimestamp = round?["timestamp"]?.ToString() ?? "";
                    string roundWinner = round?["winner"]?.ToString() ?? "";

                    foreach (var action in round?["action_events"] as JArray ?? new JArray())
                    {
                        var row = new Dictionary<string, string>
                        {
                            ["game_index"] = (gameIndex + 1).ToString(),
                            ["game_winner"] = gameWinner == "Draw" ? "2" : gameWinner == "Left" ? "0" : "1",
                            ["game_timestamp"] = gameTimestamp,
                            ["round_index"] = roundIndex.ToString(),
                            ["round_winner"] = gameWinner == "Draw" ? "2" : gameWinner == "Left" ? "0" : "1",
                            ["round_timestamp"] = roundTimestamp,
                            ["logged_at"] = action?["logged_at"]?.ToString(),
                            ["started_at"] = action?["started_at"]?.ToString(),
                            ["updated_at"] = action?["updated_at"]?.ToString(),
                            ["actor"] = action?["actor"]?.ToString() == "Left" ? "0" : "1",
                            ["target"] = action?["target"]?.ToString() == "Left" ? "0" : "1",
                            ["category"] = action?["category"]?.ToString(),
                            ["isStart"] = ((bool?)action?["isStart"] == true ? 1 : 0).ToString(),
                            ["type"] = action?["data"]?["type"]?.ToString(),
                            ["parameter"] = action?["data"]?["parameter"]?.ToString(),
                            ["duration"] = action?["data"]?["duration"]?.ToString(),
                            ["reason"] = action?["data"]?["reason"]?.ToString()
                        };

                        var before = action?["data"]?["before"];
                        if (before != null)
                        {
                            row["b_pos_x"] = before["position"]?["x"]?.ToString();
                            row["b_pos_y"] = before["position"]?["y"]?.ToString();
                            row["b_linv_x"] = before["linear_velocity"]?["x"]?.ToString();
                            row["b_linv_y"] = before["linear_velocity"]?["y"]?.ToString();
                            row["b_angv"] = before["angular_velocity"]?.ToString();
                            row["b_rot"] = before["rotation"]?["z"]?.ToString();
                        }

                        var after = action?["data"]?["after"];
                        if (after != null)
                        {
                            row["a_pos_x"] = after["position"]?["x"]?.ToString();
                            row["a_pos_y"] = after["position"]?["y"]?.ToString();
                            row["a_linv_x"] = after["linear_velocity"]?["x"]?.ToString();
                            row["a_linv_y"] = after["linear_velocity"]?["y"]?.ToString();
                            row["a_angv"] = after["angular_velocity"]?.ToString();
                            row["a_rot"] = after["rotation"]?["z"]?.ToString();
                        }

                        if (action?["category"]?.ToString() == "collision")
                        {
                            var actorData = action["data"]?["actor"];
                            var targetData = action["data"]?["target"];

                            // Actor
                            row["act_impact"] = actorData?["impact"]?.ToString();
                            row["act_linv_x"] = actorData?["linear_velocity"]?["x"]?.ToString();
                            row["act_linv_y"] = actorData?["linear_velocity"]?["y"]?.ToString();
                            row["act_angv"] = actorData?["angular_velocity"]?.ToString();
                            row["act_rot"] = actorData?["rotation"]?.ToString();
                            row["act_lock_dur"] = actorData?["lock_duration"]?.ToString();

                            // Target
                            row["tar_impact"] = targetData?["impact"]?.ToString();
                            row["tar_linv_x"] = targetData?["linear_velocity"]?["x"]?.ToString();
                            row["tar_linv_y"] = targetData?["linear_velocity"]?["y"]?.ToString();
                            row["tar_angv"] = targetData?["angular_velocity"]?.ToString();
                            row["tar_rot"] = targetData?["rotation"]?.ToString();
                            row["tar_lock_dur"] = targetData?["lock_duration"]?.ToString();
                        }

                        csvRows.Add(row);
                    }
                }
            }

            var allKeys = csvRows.SelectMany(d => d.Keys).Distinct().ToList();

            using (var writer = new StreamWriter(outputPath))
            {
                writer.WriteLine(string.Join(",", allKeys));
                foreach (var row in csvRows)
                {
                    var values = allKeys.Select(k => EscapeCsv(row.ContainsKey(k) ? row[k] ?? "" : ""));
                    writer.WriteLine(string.Join(",", values));
                }
            }

            EditorUtility.DisplayDialog("Success", "CSV generated successfully!", "OK");
            Debug.Log("CSV saved to: " + outputPath);
        }
        catch (Exception ex)
        {
            Debug.LogError("Error: " + ex.Message);
            EditorUtility.DisplayDialog("Error", "Failed to generate CSV.\n" + ex.Message, "OK");
        }
    }

    private static string EscapeCsv(string input)
    {
        if (input.Contains(",") || input.Contains("\"") || input.Contains("\n"))
        {
            return "\"" + input.Replace("\"", "\"\"") + "\"";
        }
        return input;
    }
}
#endif