#if UNITY_EDITOR
namespace SumoEditor
{
    using System.IO;
    using System.Linq;
    using System.Collections.Generic;
    using UnityEditor;
    using UnityEngine;
    using logger = Logger;
    using Newtonsoft.Json.Linq;
    using System.Text.RegularExpressions;
    using System;

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

                                ["StartedAt"] = eventLog?["StartedAt"]?.ToString(),
                                ["UpdatedAt"] = eventLog?["UpdatedAt"]?.ToString(),
                                ["Actor"] = eventLog?["Actor"]?.ToString() == "Left" ? "0" : "1",

                                ["Target"] = eventLog?["Target"]?.ToString() == "" ? "" : eventLog?["Target"]?.ToString() == "Left" ? "0" : "1",
                                
                                ["Category"] = eventLog?["Category"]?.ToString(),
                                ["State"] = eventLog?["State"].ToString(),
                            };

                            var act = eventLog?["Data"];
                            if (act != null)
                            {
                                row["Name"] = act["Name"]?.ToString();
                                row["Duration"] = act["Duration"]?.ToString();
                                row["Reason"] = act["Reason"]?.ToString();

                                var robot = act?["Robot"];
                                if (robot != null)
                                {
                                    row["BotPosX"] = robot["Position"]?["X"]?.ToString();
                                    row["BotPosY"] = robot["Position"]?["Y"]?.ToString();
                                    row["BotLinv"] = robot["LinearVelocity"]?.ToString();
                                    row["BotAngv"] = robot["AngularVelocity"]?.ToString();
                                    row["BotRot"] = robot["Rotation"]?.ToString();
                                    row["BotIsDashActive"] = (bool)robot["IsDashActive"] == true ? "1" : "0";
                                    row["BotIsSkillActive"] = (bool)robot["IsSkillActive"] == true ? "1" : "0";
                                    row["BotIsOutFromArena"] = (bool)robot["IsOutFromArena"] == true ? "1" : "0";
                                }

                                var enemyRobot = act?["EnemyRobot"];
                                if (enemyRobot != null)
                                {
                                    row["EnemyBotPosX"] = enemyRobot["Position"]?["X"]?.ToString();
                                    row["EnemyBotPosY"] = enemyRobot["Position"]?["Y"]?.ToString();
                                    row["EnemyBotLinv"] = enemyRobot["LinearVelocity"]?.ToString();
                                    row["EnemyBotAngv"] = enemyRobot["AngularVelocity"]?.ToString();
                                    row["EnemyBotRot"] = enemyRobot["Rotation"]?.ToString();
                                    row["EnemyBotIsDashActive"] = (bool)enemyRobot["IsDashActive"] == true ? "1" : "0";
                                    row["EnemyBotIsSkillActive"] = (bool)enemyRobot["IsSkillActive"] == true ? "1" : "0";
                                    row["EnemyBotIsOutFromArena"] = (bool)enemyRobot["IsOutFromArena"] == true ? "1" : "0";
                                }
                            }

                            if (eventLog?["Category"]?.ToString() == "Collision")
                            {
                                var collisionData = eventLog["Data"];

                                // Actor
                                row["ColActor"] = collisionData?["IsActor"]?.ToString();
                                row["ColImpact"] = collisionData?["Impact"]?.ToString();
                                row["ColTieBreaker"] = collisionData?["IsTieBreaker"]?.ToString();
                                row["ColLockDuration"] = collisionData?["LockDuration"]?.ToString();

                                var colRobot = collisionData?["Robot"];
                                if (colRobot != null)
                                {
                                    row["ColBotPosX"] = colRobot["Position"]?["X"]?.ToString();
                                    row["ColBotPosY"] = colRobot["Position"]?["Y"]?.ToString();
                                    row["ColBotLinv"] = colRobot["LinearVelocity"]?.ToString();
                                    row["ColBotAngv"] = colRobot["AngularVelocity"]?.ToString();
                                    row["ColBotRot"] = colRobot["Rotation"]?.ToString();
                                    row["ColBotIsDashActive"] = (bool)colRobot["IsDashActive"] == true ? "1" : "0";
                                    row["ColBotIsSkillActive"] = (bool)colRobot["IsSkillActive"] == true ? "1" : "0";
                                    row["ColBotIsOutFromArena"] = (bool)colRobot["IsOutFromArena"] == true ? "1" : "0";
                                }

                                var colEnemyRobot = collisionData?["EnemyRobot"];
                                if (colEnemyRobot != null)
                                {
                                    row["ColEnemyBotPosX"] = colEnemyRobot["Position"]?["X"]?.ToString();
                                    row["ColEnemyBotPosY"] = colEnemyRobot["Position"]?["Y"]?.ToString();
                                    row["ColEnemyBotLinv"] = colEnemyRobot["LinearVelocity"]?.ToString();
                                    row["ColEnemyBotAngv"] = colEnemyRobot["AngularVelocity"]?.ToString();
                                    row["ColEnemyBotRot"] = colEnemyRobot["Rotation"]?.ToString();
                                    row["ColEnemyBotIsDashActive"] = (bool)colEnemyRobot["IsDashActive"] == true ? "1" : "0";
                                    row["ColEnemyBotIsSkillActive"] = (bool)colEnemyRobot["IsSkillActive"] == true ? "1" : "0";
                                    row["ColEnemyBotIsOutFromArena"] = (bool)colEnemyRobot["IsOutFromArena"] == true ? "1" : "0";
                                }

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
                logger.Info("CSV saved to: " + outputPath);
            }
            catch (Exception ex)
            {
                logger.Error("Error: " + ex.Message);
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
}
#endif