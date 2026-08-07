using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using SumoManager;
using static SumoManager.LogManager;

/// <summary>
/// Drill-down folder browser shown in Replay.unity when the scene is entered from the Main Menu
/// (i.e. GameManager.ShowReplay is false, so there's no live battle log to fall back on). Walks
/// Application.persistentDataPath/Logs, descending through Single's flat folders and Batch's
/// nested checkpoint/matchup/config folders alike - a folder is a pickable replay once it
/// contains metadata.json, otherwise it's just another folder to drill into.
///
/// This script only drives pre-built UI - build PanelRoot/BackButton/MenuButton/BreadcrumbText
/// and a RowPrefab (with a ReplayPickerRow component) in the Editor with your own sprites, then
/// wire them up in the Inspector.
/// </summary>
public class ReplayPicker : MonoBehaviour
{
    public static ReplayPicker Instance { get; private set; }

    [Header("Panel")]
    public GameObject PanelRoot;
    public TMP_Text BreadcrumbText;
    public Button BackButton;
    public Button MenuButton;
    public Button OpenButton;
    public TMP_Text ReplayDetail;

    [Header("List")]
    public Transform Content;
    public GameObject RowPrefab;

    private string rootLogPath;
    private string currentPath;
    private string selectedReplayFolder;
    private Action<string> onReplaySelected;
    private readonly List<GameObject> spawnedRows = new();

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;

        rootLogPath = Path.Combine(Application.persistentDataPath, "Logs");

        if (BackButton != null)
            BackButton.onClick.AddListener(GoBack);

        if (MenuButton != null)
            MenuButton.onClick.AddListener(BackToMenu);

        if (OpenButton != null)
            OpenButton.onClick.AddListener(OpenSelectedReplay);
    }

    public void Show(Action<string> onSelect)
    {
        onReplaySelected = onSelect;

        if (PanelRoot != null)
            PanelRoot.SetActive(true);

        Navigate(rootLogPath);
    }

    public void Hide()
    {
        if (PanelRoot != null)
            PanelRoot.SetActive(false);
    }

    #region Navigation
    private void Navigate(string path)
    {
        currentPath = path;
        ClearRows();
        ClearSelection();

        if (BreadcrumbText != null)
        {
            string relative = path.Length > rootLogPath.Length
                ? path.Substring(rootLogPath.Length).TrimStart('/', '\\')
                : "";
            BreadcrumbText.text = string.IsNullOrEmpty(relative) ? "Logs" : $"Logs / {relative.Replace('\\', '/').Replace("/", " / ")}";
        }

        if (BackButton != null)
            BackButton.gameObject.SetActive(path != rootLogPath);

        string[] subDirs = Directory.Exists(path)
            ? Directory.GetDirectories(path).OrderByDescending(d => d).ToArray()
            : Array.Empty<string>();

        if (subDirs.Length == 0)
        {
            AddRow("No replays found", null);
            return;
        }

        foreach (string dir in subDirs)
        {
            bool isLeaf = File.Exists(Path.Combine(dir, "metadata.json"));
            string capturedDir = dir;

            if (isLeaf)
            {
                AddRow(BuildReplaySummary(dir), () => SelectReplay(capturedDir));
            }
            else
            {
                AddRow(Path.GetFileName(dir), () => Navigate(capturedDir));
            }
        }
    }

    private void GoBack()
    {
        if (currentPath == rootLogPath)
            return;

        string parent = Directory.GetParent(currentPath)?.FullName ?? rootLogPath;
        Navigate(parent);
    }

    // Clicking a row only selects it (shows its detail, arms OpenButton) - OpenButton is what
    // actually commits and hands off to onReplaySelected, so the user can review the detail
    // panel before deciding to play it.
    private void SelectReplay(string folder)
    {
        selectedReplayFolder = folder;

        (string detail, int gameCount) = BuildReplayDetail(folder);

        if (ReplayDetail != null)
            ReplayDetail.text = detail;

        // A config folder can exist with metadata.json but zero completed games (e.g. the run
        // was interrupted right after InitBattle wrote it) - nothing to actually play back then.
        if (OpenButton != null)
            OpenButton.interactable = gameCount > 0;
    }

    private void ClearSelection()
    {
        selectedReplayFolder = null;

        if (ReplayDetail != null)
            ReplayDetail.text = "";

        if (OpenButton != null)
            OpenButton.interactable = false;
    }

    private void OpenSelectedReplay()
    {
        if (selectedReplayFolder == null)
            return;

        Hide();
        onReplaySelected?.Invoke(selectedReplayFolder);
    }

    private string BuildReplaySummary(string folder)
    {
        try
        {
            string json = File.ReadAllText(Path.Combine(folder, "metadata.json"));
            BattleLog battleLog = JsonConvert.DeserializeObject<BattleLog>(json);

            string left = string.IsNullOrEmpty(battleLog.LeftPlayerStats?.Bot) ? "?" : battleLog.LeftPlayerStats.Bot;
            string right = string.IsNullOrEmpty(battleLog.RightPlayerStats?.Bot) ? "?" : battleLog.RightPlayerStats.Bot;
            string round = Enum.IsDefined(typeof(RoundSystem), battleLog.RoundType)
                ? ((RoundSystem)battleLog.RoundType).ToString()
                : battleLog.RoundType.ToString();

            string summary = $"{left} vs {right}  •  {round}  •  T{battleLog.BattleTime:0.#}s / AI{battleLog.ActionInterval:0.##}s";

            // Batch config folders hash away the Target/Constraint pair (see
            // BattleSimulator.GetFolderStructure) - surface it here since it's no longer readable
            // from the path itself.
            if (!string.IsNullOrEmpty(battleLog.PacingTargetFileName) || !string.IsNullOrEmpty(battleLog.PacingConstraintFileName))
                summary += $"  •  {battleLog.PacingTargetFileName} / {battleLog.PacingConstraintFileName}";

            return summary;
        }
        catch (Exception e)
        {
            Logger.Error($"[ReplayPicker] Failed to parse metadata at {folder}: {e.Message}");
            return Path.GetFileName(folder);
        }
    }

    // Folder names encode fields as "Key_value" segments joined by "__", e.g.
    // "Timer_30__ActInterval_0.1__Round_BestOf3__SkillLeft_Boost__SkillRight_Boost". Values may
    // themselves contain underscores (e.g. a Target file named "step_increase_0.0_to_1.0"), so
    // only the first "_" in each segment is treated as the key/value separator.
    private static Dictionary<string, string> ParseFolderName(string folderName)
    {
        var fields = new Dictionary<string, string>();
        foreach (string segment in folderName.Split(new[] { "__" }, StringSplitOptions.RemoveEmptyEntries))
        {
            int separator = segment.IndexOf('_');
            if (separator <= 0 || separator == segment.Length - 1)
                continue;

            fields[segment.Substring(0, separator)] = segment.Substring(separator + 1);
        }
        return fields;
    }

    // Full multi-line breakdown for the ReplayDetail panel, plus the completed game count so the
    // caller can gate OpenButton on it. metadata.json is the primary source (it's written before
    // any games run, so every leaf folder has one) - folder-name parsing only fills in fields
    // metadata.json is missing or fails to provide (e.g. a corrupted file).
    private (string detail, int gameCount) BuildReplayDetail(string folder)
    {
        Dictionary<string, string> parsed = ParseFolderName(Path.GetFileName(folder));
        BattleLog battleLog = null;

        string metadataPath = Path.Combine(folder, "metadata.json");
        if (File.Exists(metadataPath))
        {
            try
            {
                battleLog = JsonConvert.DeserializeObject<BattleLog>(File.ReadAllText(metadataPath));
            }
            catch (Exception e)
            {
                Logger.Error($"[ReplayPicker] Failed to parse metadata at {folder}: {e.Message}");
            }
        }

        string Field(string metaValue, string parsedKey) =>
            !string.IsNullOrEmpty(metaValue) ? metaValue : parsed.GetValueOrDefault(parsedKey, "-");

        string left = Field(battleLog?.LeftPlayerStats?.Bot, "Left");
        string right = Field(battleLog?.RightPlayerStats?.Bot, "Right");
        string timer = Field(battleLog != null ? battleLog.BattleTime.ToString("0.#") : null, "Timer");
        string actionInterval = Field(battleLog != null ? battleLog.ActionInterval.ToString("0.##") : null, "ActInterval");
        string round = Field(battleLog != null && Enum.IsDefined(typeof(RoundSystem), battleLog.RoundType)
            ? ((RoundSystem)battleLog.RoundType).ToString()
            : null, "Round");
        string skillLeft = Field(battleLog?.LeftPlayerStats?.SkillType, "SkillLeft");
        string skillRight = Field(battleLog?.RightPlayerStats?.SkillType, "SkillRight");
        string target = Field(battleLog?.PacingTargetFileName, "Target");
        string constraint = Field(battleLog?.PacingConstraintFileName, "Constraint");
        string segmentDur = Field(battleLog?.PacingSegmentDuration.ToString(), "Constraint");
        int gameCount = Directory.GetFiles(folder, "game_*.json").Length;

        var lines = new List<string>
        {
            $"Left: {left}",
            $"Right: {right}",
            $"Timer: {timer}",
            $"Action Interval: {actionInterval}",
            $"Round: {round}",
            $"Skill Left: {skillLeft}",
            $"Skill Right: {skillRight}",
        };

        if (target != "-" || constraint != "-")
        {
            lines.Add($"Pac Target: {target}");
            lines.Add($"Pac Constraint: {constraint}");
            lines.Add($"Pac Duration: {segmentDur}");
        }

        lines.Add($"Games: {gameCount}");

        if (battleLog != null)
            lines.Add($"Created: {DateTimeOffset.FromUnixTimeSeconds(battleLog.CreatedAt).LocalDateTime:yyyy-MM-dd HH:mm}");

        return (string.Join("\n", lines), gameCount);
    }

    public void BackToMenu()
    {
        SceneManager.LoadScene("MainMenu");
    }
    #endregion

    #region Row spawning
    private void AddRow(string label, Action onClick)
    {
        if (RowPrefab == null || Content == null)
        {
            Logger.Error("[ReplayPicker] RowPrefab/Content not assigned in the inspector.");
            return;
        }

        GameObject rowGO = Instantiate(RowPrefab, Content);
        spawnedRows.Add(rowGO);

        Button row = rowGO.GetComponent<Button>();
        TMP_Text txt = rowGO.GetComponentInChildren<TMP_Text>();
        if (row == null)
        {
            Logger.Error("[ReplayPicker] RowPrefab is missing a ReplayPickerRow component.");
            return;
        }

        if (txt.text != null)
            txt.text = label;

        if (row != null)
        {
            row.interactable = onClick != null;
            if (onClick != null)
                row.onClick.AddListener(() => onClick());
        }
    }

    private void ClearRows()
    {
        foreach (GameObject row in spawnedRows)
            if (row != null)
                Destroy(row);

        spawnedRows.Clear();
    }
    #endregion
}
