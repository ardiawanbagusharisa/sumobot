using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using Newtonsoft.Json;
using UnityEngine;

namespace SumoLeaderboard
{
    /// <summary>
    /// Persists LeaderboardData as JSON under Application.persistentDataPath,
    /// following the same pattern as LogManager. Writes are atomic
    /// (temp file + move) and the payload carries a tamper-evidence checksum.
    /// A corrupt file is backed up and replaced with a fresh one (FR-6).
    /// </summary>
    public class LeaderboardStore : ILeaderboardStore
    {
        public const string FolderName = "Leaderboards";
        public const string FileName = "leaderboards.json";
        private const string ChecksumSalt = "sumobot-leaderboard-v1";

        private readonly string folderPath;
        private readonly string filePath;

        public LeaderboardStore(string rootPath = null)
        {
            folderPath = Path.Combine(rootPath ?? Application.persistentDataPath, FolderName);
            filePath = Path.Combine(folderPath, FileName);
        }

        public string FilePath => filePath;

        public LeaderboardData Load()
        {
            try
            {
                if (!File.Exists(filePath))
                    return new LeaderboardData();

                string json = File.ReadAllText(filePath);
                LeaderboardData data = JsonConvert.DeserializeObject<LeaderboardData>(json);

                if (data == null)
                    throw new InvalidDataException("Leaderboard file deserialized to null.");

                // Older schema: the stored checksum was computed over the old
                // shape, so it cannot match after deserialization gains new
                // fields. Migrate instead of treating it as corruption.
                if (data.Version < LeaderboardData.CurrentVersion)
                {
                    Migrate(data);
                    Save(data); // reseal with a current-schema checksum
                    return data;
                }

                string expected = data.Checksum;
                data.Checksum = null;
                string actual = ComputeChecksum(data);

                if (!string.IsNullOrEmpty(expected) && expected != actual)
                {
                    Logger.Error("[LeaderboardStore] Checksum mismatch, resetting leaderboards.");
                    BackupCorruptFile();
                    return new LeaderboardData();
                }

                return data;
            }
            catch (Exception ex)
            {
                Logger.Error($"[LeaderboardStore] Failed to load ({ex.Message}), resetting leaderboards.");
                BackupCorruptFile();
                return new LeaderboardData();
            }
        }

        public void Save(LeaderboardData data)
        {
            try
            {
                Directory.CreateDirectory(folderPath);

                data.Checksum = null;
                data.Checksum = ComputeChecksum(data);

                string json = JsonConvert.SerializeObject(data, Formatting.Indented);
                string tempPath = filePath + ".tmp";
                File.WriteAllText(tempPath, json);

                if (File.Exists(filePath))
                    File.Delete(filePath);
                File.Move(tempPath, filePath);
            }
            catch (Exception ex)
            {
                Logger.Error($"[LeaderboardStore] Failed to save: {ex.Message}");
            }
        }

        /// <summary>
        /// Upgrades older payloads in place. v1 → v2: tables gained a GameMode
        /// key; every v1 table came from offline multiplayer, and enum default
        /// (0) is already Multiplayer, so only the version stamp changes.
        /// </summary>
        private static void Migrate(LeaderboardData data)
        {
            Logger.Info($"[LeaderboardStore] Migrating leaderboards v{data.Version} -> v{LeaderboardData.CurrentVersion}.");
            data.Version = LeaderboardData.CurrentVersion;
        }

        private void BackupCorruptFile