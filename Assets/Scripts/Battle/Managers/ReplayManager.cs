using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Newtonsoft.Json;
using static SumoManager.LogManager;
using SumoLog;
using TMPro;
using System;
using System.Collections;
using UnityEngine.UI;
using SumoCore;

public class ReplayManager : MonoBehaviour
{
    public static ReplayManager Instance { get; private set; }

    #region Replay Configuration properties
    [Header("Replay Configuration")]
    public bool IsEnable = false;

    [Range(0f, 5f)]
    public float playbackSpeed = 1f;
    public bool autoStart = true;
    public SumoController leftPlayer;
    public SumoController rightPlayer;
    #endregion

    [Header("Replay Configuration")]
    public GameObject mainPanel;

    #region Replay control properties
    [Header("Replay Controls")]
    public Slider TimeSliderUI;
    public TMP_Text TimeLabel;
    public Button PreviousGameButton;
    public Button NextGameButton;
    public Button PreviousRoundButton;
    public Button NextRoundButton;
    public Slider PlaybackSpeedSlider;
    public TMP_Text PlaybackSpeedLabel;
    #endregion

    #region Replay UI properties
    [Header("Replay UI")]
    public TMP_Text GameUI;
    public TMP_Text RoundUI;
    public TMP_Text GameDurationUI;
    public TMP_Text GameBestOf;

    public TMP_Text LeftBotName;
    public TMP_Text LeftSkillType;
    public TMP_Text LeftWinCount;
    public TMP_Text LeftActionTaken;

    public TMP_Text RightBotName;
    public TMP_Text RightWinCount;
    public TMP_Text RightSkillType;
    public TMP_Text RightActionTaken;
    #endregion

    #region Runtime (readonly) properties 
    private readonly List<GameLog> gameLogs = new();
    private int currentGameIndex = 0;
    private int currentRoundIndex = 0;
    private float currentTime = 0f;
    private bool isPlaying = false;

    private List<EventLog> currentRoundEvents = new();
    private float currentRoundDuration = 0f;
    private int leftEventIndex = 0;
    private int rightEventIndex = 0;
    private bool isDraggingSlider = false;
    private BattleLog metadata;
    private readonly Dictionary<string, EventLog> leftEventsMap = new();
    private readonly Dictionary<string, EventLog> rightEventsMap = new();
    private readonly List<EventLog> leftEvents = new();
    private readonly List<EventLog> rightEvents = new();
    #endregion

    #region Unity methods
    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }

    void Start()
    {
        if (!IsEnable)
            return;

        mainPanel.SetActive(true);

        string basePath = Path.Combine(Application.persistentDataPath, "Logs");
        string folder = EditorUtility.OpenFolderPanel("Select Replay Folder", basePath, "");

        if (string.IsNullOrEmpty(folder)) return;

        LoadAllGameLogs(folder);
        LoadMetadata(folder);
        LoadRound(currentGameIndex, currentRoundIndex);

        leftPlayer.Side = PlayerSide.Left;
        rightPlayer.Side = PlayerSide.Right;
        leftPlayer.UpdateDirectionColor();
        rightPlayer.UpdateDirectionColor();

        if (autoStart)
            isPlaying = true;

        PreviousGameButton?.onClick.AddListener(GoToPreviousGame);
        NextGameButton?.onClick.AddListener(GoToNextGame);
        PreviousRoundButton?.onClick.AddListener(GoToPreviousRound);
        NextRoundButton?.onClick.AddListener(GoToNextRound);

        // Playback speed slider
        if (PlaybackSpeedSlider != null)
        {
            PlaybackSpeedSlider.minValue = 0f;
            PlaybackSpeedSlider.maxValue = 5f;
            PlaybackSpeedSlider.value = playbackSpeed;

            PlaybackSpeedSlider.onValueChanged.AddListener(value =>
            {
                playbackSpeed = value;
                if (PlaybackSpeedLabel != null)
                    PlaybackSpeedLabel.text = $"Playback Speed: {value:0.#}x";
            });

            if (PlaybackSpeedLabel != null)
                PlaybackSpeedLabel.text = $"Playback Speed: {playbackSpeed:0.#}x";
        }
    }

    void OnEnable()
    {
        if (TimeSliderUI != null)
        {
            TimeSliderUI.onValueChanged.AddListener(OnTimeSliderChanged);
        }
    }

    void OnDisable()
    {
        if (TimeSliderUI != null)
        {
            TimeSliderUI.onValueChanged.RemoveListener(OnTimeSliderChanged);
        }
    }


    void Update()
    {
        if (!isPlaying || !IsEnable) return;

        currentTime += Time.deltaTime * playbackSpeed;

        if (TimeSliderUI != null && !isDraggingSlider)
            TimeSliderUI.value = Mathf.Ceil(currentTime);

        if (TimeLabel != null)
            TimeLabel.text = $"{FormatTime(currentTime)} / {FormatTime(currentRoundDuration)}";

        InterpolateBot(leftPlayer, leftEvents, ref leftEventIndex);
        InterpolateBot(rightPlayer, rightEvents, ref rightEventIndex);

        DisplayCurrentEventInfo();

        if (currentTime > currentRoundDuration)
        {
            currentRoundIndex++;

            if (currentRoundIndex >= gameLogs[currentGameIndex].Rounds.Count)
            {
                leftEventsMap.Clear();
                rightEventsMap.Clear();
                metadata.LeftPlayerStats.ActionTaken = 0;
                metadata.LeftPlayerStats.WinPerGame = 0;
                metadata.RightPlayerStats.ActionTaken = 0;
                metadata.RightPlayerStats.WinPerGame = 0;

                currentGameIndex++;
                currentRoundIndex = 0;

                if (currentGameIndex >= gameLogs.Count)
                {
                    isPlaying = false;
                    Debug.Log("Replay finished.");
                    return;
                }
            }
            LoadRound(currentGameIndex, currentRoundIndex);
        }
    }
    #endregion

    #region Core Logics
    void LoadRound(int gameIdx, int roundIdx)
    {
        currentTime = 0f;
        currentRoundEvents.Clear();
        leftEvents.Clear();
        rightEvents.Clear();
        ResetMetadata();

        var round = gameLogs[gameIdx].Rounds[roundIdx];

        currentRoundEvents = round.PlayerEvents.OrderBy(e => e.UpdatedAt).ToList();

        var left = currentRoundEvents.Where(x => x.Actor == "Left").ToList();
        var right = currentRoundEvents.Where(x => x.Actor == "Right").ToList();

        for (int i = 0; i < left.Count - 1; i++)
        {
            leftEvents.Add(left[i]);
        }
        for (int i = 0; i < right.Count - 1; i++)
        {
            rightEvents.Add(right[i]);
        }

        currentRoundDuration = currentRoundEvents.Max(e => e.UpdatedAt);

        if (TimeSliderUI != null)
        {
            TimeSliderUI.minValue = 0f;
            TimeSliderUI.maxValue = Mathf.Ceil(currentRoundDuration);
            TimeSliderUI.value = 0f;
        }

        leftEventIndex = 0;
        rightEventIndex = 0;
    }

    void InterpolateBot(SumoController controller, List<EventLog> events, ref int index)
    {
        if (index >= events.Count)
            return;

        EventLog currentEvent = events[index];
        EventLog nextEvent;

        if (index == events.Count - 1)
        {
            currentEvent = events[index - 1];
            nextEvent = events[index];
        }
        else
        {
            nextEvent = events[index + 1];
        }

        if (currentTime > nextEvent.UpdatedAt)
        {
            index++;
            if (index >= events.Count)
                return;

            currentEvent = events[index];
        }

        if (currentEvent.Category == "Action")
        {
            if (currentEvent.Actor == "Left")
            {
                string key = $"Action_{currentEvent.Data["Name"]}_{currentEvent.Data?["Parameter"] ?? null}_{currentEvent.StartedAt}";
                if (!leftEventsMap.ContainsKey(key))
                    leftEventsMap.Add(key, currentEvent);
            }
            else if (currentEvent.Actor == "Right")
            {
                string key = $"Action_{currentEvent.Data["Name"]}_{currentEvent.Data?["Parameter"] ?? null}_{currentEvent.StartedAt}";
                if (!rightEventsMap.ContainsKey(key))
                    rightEventsMap.Add(key, currentEvent);
            }
        }

        float t = Mathf.InverseLerp(
            currentEvent.UpdatedAt,
            nextEvent.UpdatedAt,
            currentTime
        );

        if (t >= 0.0f && t <= 1.0)
        {
            var start = BaseLog.FromMap(currentEvent.Data);
            var end = BaseLog.FromMap(nextEvent.Data);

            controller.transform.position = Vector3.Lerp(start.Position, end.Position, t);
            controller.transform.rotation = Quaternion.Lerp(
                Quaternion.Euler(0, 0, start.Rotation),
                Quaternion.Euler(0, 0, end.Rotation),
                t
            );
        }
    }


    void LoadAllGameLogs(string folderPath)
    {
        string[] files = Directory.GetFiles(folderPath, "game_*.json");
        foreach (var file in files)
        {
            string json = File.ReadAllText(file);
            var log = JsonConvert.DeserializeObject<GameLog>(json);
            gameLogs.Add(log);
        }
    }
    void LoadMetadata(string folderPath)
    {
        string[] metadataFile = Directory.GetFiles(folderPath, "metadata.json");
        string json = File.ReadAllText(metadataFile[0]);
        metadata = JsonConvert.DeserializeObject<BattleLog>(json);

        metadata.LeftPlayerStats.WinPerGame = 0;
        metadata.RightPlayerStats.WinPerGame = 0;
        metadata.LeftPlayerStats.ActionTaken = 0;
        metadata.RightPlayerStats.ActionTaken = 0;
    }

    void DisplayCurrentEventInfo()
    {
        var current = currentRoundEvents.LastOrDefault(e => e.UpdatedAt <= currentTime);
        if (current != null)
        {
            GameUI.SetText($"Game {currentGameIndex + 1}");
            RoundUI.SetText($"Round {currentRoundIndex + 1}");
            GameDurationUI.SetText($"Duration: {metadata.BattleTime}");
            GameBestOf.SetText($"Best Of: {metadata.RoundType}");

            metadata.LeftPlayerStats.ActionTaken = leftEventsMap.Count;
            metadata.RightPlayerStats.ActionTaken = rightEventsMap.Count;

            LeftBotName.SetText(metadata.LeftPlayerStats.Bot);
            LeftSkillType.SetText(metadata.LeftPlayerStats.SkillType);
            LeftWinCount.SetText(metadata.LeftPlayerStats.WinPerGame.ToString());
            LeftActionTaken.SetText(metadata.LeftPlayerStats.ActionTaken.ToString());

            RightBotName.SetText(metadata.RightPlayerStats.Bot);
            RightSkillType.SetText(metadata.RightPlayerStats.SkillType);
            RightWinCount.SetText(metadata.RightPlayerStats.WinPerGame.ToString());
            RightActionTaken.SetText(metadata.RightPlayerStats.ActionTaken.ToString());
        }
    }
    #endregion

    #region Control Logic

    void GoToPreviousGame()
    {
        if (currentGameIndex > 0)
        {
            currentGameIndex--;
            currentRoundIndex = 0;
            LoadRound(currentGameIndex, currentRoundIndex);
            isPlaying = true;
        }
    }

    void GoToNextGame()
    {
        if (currentGameIndex < gameLogs.Count - 1)
        {
            currentGameIndex++;
            currentRoundIndex = 0;
            LoadRound(currentGameIndex, currentRoundIndex);
            isPlaying = true;
        }
    }

    void GoToPreviousRound()
    {
        if (currentRoundIndex > 0)
        {
            currentRoundIndex--;
            LoadRound(currentGameIndex, currentRoundIndex);
            isPlaying = true;
        }
        else if (currentGameIndex > 0)
        {
            currentGameIndex--;
            currentRoundIndex = gameLogs[currentGameIndex].Rounds.Count - 1;
            LoadRound(currentGameIndex, currentRoundIndex);
            isPlaying = true;
        }
    }

    void GoToNextRound()
    {
        if (currentRoundIndex < gameLogs[currentGameIndex].Rounds.Count - 1)
        {
            currentRoundIndex++;
            LoadRound(currentGameIndex, currentRoundIndex);
        }
        else if (currentGameIndex < gameLogs.Count - 1)
        {
            currentGameIndex++;
            currentRoundIndex = 0;
            LoadRound(currentGameIndex, currentRoundIndex);
        }
    }

    void OnTimeSliderChanged(float value)
    {
        if (isDraggingSlider)
        {
            currentTime = value;
            ResetMetadata();
        }
    }
    void ResetMetadata()
    {
        leftEventIndex = 0;
        rightEventIndex = 0;
        leftEventsMap.Clear();
        rightEventsMap.Clear();
        metadata.LeftPlayerStats.ActionTaken = 0;
        metadata.LeftPlayerStats.WinPerGame = 0;
        metadata.RightPlayerStats.ActionTaken = 0;
        metadata.RightPlayerStats.WinPerGame = 0;
    }

    public void OnTimeSliderPointerDown()
    {
        isDraggingSlider = true;
    }

    public void OnTimeSliderPointerUp()
    {
        isDraggingSlider = false;

        isPlaying = false;

        currentTime = TimeSliderUI.value;

        ResetMetadata();

        InterpolateBot(leftPlayer, leftEvents, ref leftEventIndex);
        InterpolateBot(rightPlayer, rightEvents, ref rightEventIndex);

        StartCoroutine(ResumeAfterLoadRound());
    }

    private IEnumerator ResumeAfterLoadRound()
    {
        yield return null;

        isPlaying = true;
    }

    #endregion


    string FormatTime(float time)
    {
        int minutes = Mathf.FloorToInt(time / 60f);
        int seconds = Mathf.CeilToInt(time % 60f);
        return $"{minutes:00}:{seconds:00}";
    }


}