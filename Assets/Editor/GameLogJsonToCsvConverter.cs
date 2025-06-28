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

                int gameIndex = (int?)root["Index"] ?? -1;
                string gameTimestamp = (string)root["Timestamp"] ?? "";
                string gameWinner = root?["Winner"]?.ToString() ?? "";

                foreach (var round in root?["Rounds"] as JArray ?? new JArray())
                {
                    int roundIndex = (int?)round["Index"] ?? -1;
                    string roundTimestamp = round?["Timestamp"]?.ToString() ?? "";
                    string roundWinner = round?["Winner"]?.ToString() ?? "";

                    foreach (var eventLog in round?["PlayerEvents"] as JArray ?? new JArray())
                    {
                        var row = new Dictionary<string, string>
                        {
                            ["GameIndex"] = (gameIndex + 1).ToString(),
                            ["GameWinner"] = gameWinner == "Draw" ? "2" : gameWinner == "Left" ? "0" : "1",
                            ["GameTimestamp"] = gameTimestamp,
                            ["RoundIndex"] = roundIndex.ToString(),
                            ["RoundWinner"] = gameWinner == "Draw" ? "2" : gameWinner == "Left" ? "0" : "1",
                            ["RoundTimestamp"] = roundTimestamp,

                            ["LoggedAt"] = eventLog?["LoggedAt"]?.ToString(),
                            ["StartedAt"] = eventLog?["StartedAt"]?.ToString(),
                            ["UpdatedAt"] = eventLog?["UpdatedAt"]?.ToString(),
                            ["Actor"] = eventLog?["Actor"]?.ToString() == "Left" ? "0" : "1",
                            ["Target"] = eventLog?["Target"]?.ToString() == "Left" ? "0" : "1",
                            ["Category"] = eventLog?["Category"]?.ToString(),
                            ["IsStart"] = ((bool?)eventLog?["IsStart"] == true ? 1 : 0).ToString(),
                        };

                        var data = eventLog?["Data"]?["Robot"];
                        if (data != null)
                        {
                            row["PosX"] = data["Position"]?["X"]?.ToString();
                            row["PosY"] = data["Position"]?["Y"]?.ToString();
                            row["LinvX"] = data["LinearVelocity"]?["X"]?.ToString();
                            row["LinvY"] = data["LinearVelocity"]?["Y"]?.ToString();
                            row["Angv"] = data["AngularVelocity"]?.ToString();
                            row["Rot"] = data["Rotation"]?["Z"]?.ToString();
                        }

                        if (eventLog?["Category"]?.ToString() == "Collision")
                        {
                            var collisionData = eventLog["Data"];

                            // Actor
                            row["ColIsActor"] = collisionData?["IsActor"]?.ToString();
                            row["ColImpact"] = collisionData?["Impact"]?.ToString();
                            row["ColLockDuration"] = collisionData?["LockDuration"]?.ToString();
                            row["ColDuration"] = collisionData?["Duration"]?.ToString();
                        }
                        else if (eventLog?["Category"]?.ToString() == "Action")
                        {
                            var actionData = eventLog["Data"];

                            // Actor
                            row["ActName"] = actionData?["Name"]?.ToString();
                            row["ActParam"] = actionData?["Parameter"]?.ToString();
                            row["ActReason"] = actionData?["Reason"]?.ToString();
                            row["ActDuration"] = actionData?["Duration"]?.ToString();
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