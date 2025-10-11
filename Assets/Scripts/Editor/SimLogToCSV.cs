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

    public class SimLogToCSV : EditorWindow
    {
        [MenuItem("Tools/Simulation Log to CSV")]
        public static void ShowWindow()
        {
            GetWindow<SimLogToCSV>("JSON to CSV Converter");
        }

        private void OnGUI()
        {
            GUILayout.Label("Simulation Log JSON to CSV", EditorStyles.boldLabel);

            if (GUILayout.Button("Convert to CSV", GUILayout.Height(40)))
            {
                // ConvertAllConfigs(Path.Combine(Application.persistentDataPath, "Simulation"));
                var name = "Timer_30__ActInterval_0.5__Round_BestOf5__SkillLeft_Stone__SkillRight_Boost";
                var path = Path.Combine(Application.persistentDataPath, "Simulation", "Bot_NN_vs_Bot_UtilityAI", "Timer_30__ActInterval_0.5__Round_BestOf5__SkillLeft_Stone__SkillRight_Boost");
                ConvertLogsToCsv(path, Path.Combine(path, $"{name}.csv"));
            }
        }

        private static int ExtractGameIndex(string filename)
        {
            // e.g., game_001 -> 1
            var match = Regex.Match(filename, @"game_(\d+)");
            return match.Success ? int.Parse(match.Groups[1].Value) : -1;
        }

        public void ConvertAllConfigs(string simulationRoot)
        {
            try
            {
                // Find all config folders recursively
                var configFolders = Directory.GetDirectories(simulationRoot, "Timer_*", SearchOption.AllDirectories);
                for (var i = 0; i < configFolders.Count(); i++)
                {
                    var configFolder = configFolders[i];

                    // Derive output CSV name from folder name
                    string configName = Path.GetFileName(configFolder);
                    string parentName = Path.GetFileName(Path.GetDirectoryName(configFolder)); // e.g. BOT_A_vs_BOT_B

                    string outputPath = Path.Combine(configFolder, $"{configName}.csv");

                    logger.Info($"[SimLogToCSV] Running {i + 1}/{configFolders.Count()} -> {configName}");
                    EditorUtility.DisplayProgressBar("Converting Logs to CSV",
                    $"Processing {i + 1}/{configFolders.Count()} {configName}",
                    (float)i + 1 / configFolders.Count());

                    // Call your existing converter for this config folder
                    ConvertLogsToCsv(configFolder, outputPath);
                }

                EditorUtility.ClearProgressBar();
                EditorUtility.DisplayDialog("Success", "All CSV files generated successfully!", "OK");
            }
            catch (Exception ex)
            {
                logger.Error("Error: " + ex.Message);
                EditorUtility.ClearProgressBar();
                EditorUtility.DisplayDialog("Error", "Failed to generate CSVs.\n" + ex.Message, "OK");
            }
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
                            if (eventLog?["Category"]?.ToString() == "LastPosition") continue;

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
            }
            catch (Exception ex)
            {
                logger.Error($"Error reading config from {folderPath} : {ex.Message}, {ex.Source}");
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