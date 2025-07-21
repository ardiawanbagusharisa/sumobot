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
    public Transform leftPlayer;
    public Transform rightPlayer;
    #endregion

    [Header("Replay Configuration")]
    public bool LoadFromPath = true;

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

    #region Replay Charts
    [Header("Replay Charts")]
    public GameObject SidebarPanel;
    public ChartManager ActionPerSecondChart;
    public float ActionTimeInterval = 2f;
    public ChartManager MostActionChartLeft;
    public ChartManager MostActionChartRight;
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
    private Dictionary<string, EventLog> leftEventsMap = new();
    private Dictionary<string, EventLog> rightEventsMap = new();
    private List<EventLog> leftEvents = new();
    private List<EventLog> rightEvents = new();
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

        if (LoadFromPath)
            LoadGameFromPath();
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

        if (PlaybackSpeedSlider != null)
        {
            PlaybackSpeedSlider.onValueChanged.RemoveListener(OnPlayBackSpeedChanged);
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

            if (gameLogs[currentGameIndex].Rounds.Count - 1 == currentRoundIndex)
            {
                DisplayCurrentEventInfo();
                isPlaying = false;
                Debug.Log("Replay finished.");
                return;
            }
            else
                currentRoundIndex++;

            if (currentRoundIndex >= gameLogs[currentGameIndex].Rounds.Count)
            {
                currentGameIndex++;

                var games = gameLogs.Take(currentGameIndex).ToList();
                metadata.LeftPlayerStats.WinPerGame = games.Select((i) => i.Winner == "Left").Count();
                metadata.LeftPlayerStats.WinPerGame = games.Select((i) => i.Winner == "Right").Count();

                currentRoundIndex = 0;

                if (currentGameIndex >= gameLogs.Count)
                {
                    DisplayCurrentEventInfo();
                    isPlaying = false;
                    Debug.Log("Replay finished.");
                    return;
                }
            }
            LoadRound(currentGameIndex, currentRoundIndex);
        }
    }

    void FixedUpdate()
    {
        if (!isPlaying || !IsEnable) return;
        ShowActionChart();
        ShowMostActionChart(PlayerSide.Left);
        ShowMostActionChart(PlayerSide.Right);
    }
    #endregion

    void Init()
    {
        LoadRound(currentGameIndex, currentRoundIndex);
        if (autoStart)
            isPlaying = true;

        PreviousGameButton?.onClick.AddListener(GoToPreviousGame);
        NextGameButton?.onClick.AddListener(GoToNextGame);
        PreviousRoundButton?.onClick.AddListener(GoToPreviousRound);
        NextRoundButton?.onClick.AddListener(GoToNextRound);

        if (PlaybackSpeedSlider == null)
            return;

        PlaybackSpeedSlider.minValue = 0f;
        PlaybackSpeedSlider.maxValue = 5f;
        PlaybackSpeedSlider.value = playbackSpeed;

        PlaybackSpeedSlider.onValueChanged.AddListener(OnPlayBackSpeedChanged);

        if (PlaybackSpeedLabel != null)
            PlaybackSpeedLabel.text = $"Playback Speed: {playbackSpeed:0.#}x";
    }

    void OnPlayBackSpeedChanged(float value)
    {
        playbackSpeed = value;
        if (PlaybackSpeedLabel != null)
            PlaybackSpeedLabel.text = $"Playback Speed: {value:0.#}x";
    }

    #region Core Logics
    void LoadRound(int gameIdx, int roundIdx)
    {
        ResetReplay(includePlayer: roundIdx == 0);

        var round = gameLogs[gameIdx].Rounds[roundIdx];

        currentRoundEvents = round.PlayerEvents.OrderBy(e => e.UpdatedAt).ToList();

        leftEvents = currentRoundEvents.Where(x => x.Actor == "Left").ToList();
        rightEvents = currentRoundEvents.Where(x => x.Actor == "Right").ToList();

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

    void InterpolateBot(Transform player, List<EventLog> events, ref int index)
    {
        if (!isPlaying) return;

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
            var key = currentEvent.GetKey();

            if (currentEvent.Actor == "Left")
            {
                if (!leftEventsMap.ContainsKey(key))
                    leftEventsMap.Add(key, currentEvent);
            }
            else if (currentEvent.Actor == "Right")
            {
                if (!rightEventsMap.ContainsKey(key))
                    rightEventsMap.Add(key, currentEvent);
            }
        }

        float t = Mathf.InverseLerp(
            currentEvent.UpdatedAt,
            nextEvent.UpdatedAt,
            currentTime
        );

        var start = BaseLog.FromMap(currentEvent.Data);
        var end = BaseLog.FromMap(nextEvent.Data);

        player.transform.position = Vector3.Lerp(start.Position, end.Position, t);
        player.transform.rotation = Quaternion.Slerp(
            Quaternion.Euler(0, 0, start.Rotation),
            Quaternion.Euler(0, 0, end.Rotation),
            t
        );
    }


    public void LoadGameFromPath()
    {
        string basePath = Path.Combine(Application.persistentDataPath, "Logs");
        string folder = EditorUtility.OpenFolderPanel("Select Replay Folder", basePath, "");

        if (string.IsNullOrEmpty(folder)) return;

        string[] files = Directory.GetFiles(folder, "game_*.json");
        foreach (var file in files)
        {
            string json = File.ReadAllText(file);
            var log = JsonConvert.DeserializeObject<GameLog>(json);
            gameLogs.Add(log);
        }

        string[] metadataFile = Directory.GetFiles(folder, "metadata.json");
        string jsonMetaData = File.ReadAllText(metadataFile[0]);
        metadata = JsonConvert.DeserializeObject<BattleLog>(jsonMetaData);

        metadata.LeftPlayerStats.WinPerGame = 0;
        metadata.RightPlayerStats.WinPerGame = 0;
        metadata.LeftPlayerStats.ActionTaken = 0;
        metadata.RightPlayerStats.ActionTaken = 0;

        Init();
    }

    public void LoadGameFromBattle(BattleLog battleLog)
    {
        foreach (var game in battleLog.Games)
        {
            gameLogs.Add(game);
        }

        metadata = battleLog;

        metadata.LeftPlayerStats.WinPerGame = 0;
        metadata.RightPlayerStats.WinPerGame = 0;
        metadata.LeftPlayerStats.ActionTaken = 0;
        metadata.RightPlayerStats.ActionTaken = 0;

        Init();
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
        }

        LoadRound(currentGameIndex, currentRoundIndex);
        isPlaying = true;
    }

    void GoToNextGame()
    {
        if (currentGameIndex < gameLogs.Count - 1)
        {
            currentGameIndex++;
            currentRoundIndex = 0;
        }

        LoadRound(currentGameIndex, currentRoundIndex);
        isPlaying = true;
    }

    void GoToPreviousRound()
    {
        if (currentRoundIndex > 0)
            currentRoundIndex--;

        LoadRound(currentGameIndex, currentRoundIndex);
        isPlaying = true;
    }

    void GoToNextRound()
    {
        if (currentRoundIndex < gameLogs[currentGameIndex].Rounds.Count - 1)
            currentRoundIndex++;

        LoadRound(currentGameIndex, currentRoundIndex);
        isPlaying = true;
    }

    void OnTimeSliderChanged(float value)
    {
        if (isDraggingSlider)
        {
            currentTime = value;
            ResetReplay(includeEvents: false);
        }
    }
    void ResetReplay(bool includePlayer = true, bool includeEvents = true)
    {
        if (includeEvents)
        {
            currentRoundEvents.Clear();
            leftEvents.Clear();
            rightEvents.Clear();
        }

        currentTime = 0f;
        leftEventIndex = 0;
        rightEventIndex = 0;
        leftEventsMap.Clear();
        rightEventsMap.Clear();
        ActionPerSecondChart.ClearChartSeries();
        MostActionChartLeft.ClearChartSeries();

        if (includePlayer)
        {
            metadata.LeftPlayerStats.ActionTaken = 0;
            metadata.RightPlayerStats.ActionTaken = 0;
        }
    }

    public void OnTimeSliderPointerDown()
    {
        isDraggingSlider = true;
    }

    public void OnTimeSliderPointerUp()
    {
        isDraggingSlider = false;
        isPlaying = false;

        ResetReplay(includeEvents: false);

        currentTime = TimeSliderUI.value;

        IEnumerable<EventLog> lastLeft = leftEvents.TakeWhile((x) => x.UpdatedAt < currentTime);
        IEnumerable<EventLog> lastRight = rightEvents.TakeWhile((x) => x.UpdatedAt < currentTime);

        if (lastLeft.Count() > 0 && lastRight.Count() > 0)
        {
            leftEventIndex = leftEvents.IndexOf(lastLeft.LastOrDefault());
            rightEventIndex = rightEvents.IndexOf(lastRight.LastOrDefault());

            leftEventsMap.Clear();
            rightEventsMap.Clear();
            leftEventsMap = lastLeft.Where((x) => x.Category == "Action" && !x.IsStart).ToDictionary((x) => x.GetKey());
            rightEventsMap = lastRight.Where((x) => x.Category == "Action" && !x.IsStart).ToDictionary((x) => x.GetKey());
        }

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

    private void ShowActionChart()
    {
        int timeFrame = 1;

        Dictionary<int, (float, float)> actionTakensMap = new();

        for (float i = 0; i < currentTime; i += ActionTimeInterval)
        {
            int leftActionAmount = leftEventsMap.Values.Where((x) => x.UpdatedAt >= i && x.UpdatedAt < (i + ActionTimeInterval)).Count();

            int rightActionAmount = rightEventsMap.Values.Where((x) => x.UpdatedAt >= i && x.UpdatedAt < (i + ActionTimeInterval)).Count();

            actionTakensMap[timeFrame] = (leftActionAmount, rightActionAmount);

            timeFrame += 1;
        }

        var leftAmount = actionTakensMap.Select((x) => x.Value.Item1).ToArray();
        var rightAmount = actionTakensMap.Select((x) => x.Value.Item2).ToArray();

        ChartSeries chartLeft = new(
            $"P1_Round_{currentRoundIndex + 1}",
            leftAmount, ChartSeries.ChartType.Line, Color.green);

        ChartSeries chartRight = new(
            $"P2_Round_{currentRoundIndex + 1}",
            rightAmount, ChartSeries.ChartType.Line, Color.red);

        void setup()
        {
            ActionPerSecondChart.Setup(
                xGridSpacing: (int)ActionTimeInterval,
                onXLabelCreated: (index) =>
                {
                    if (ActionTimeInterval > 1.0f)
                    {
                        float xlabel = index * ActionTimeInterval;
                        return Mathf.Floor(xlabel).ToString("0.#");
                    }
                    return index.ToString();
                });
        }

        chartLeft.OnPrepareToDraw = setup;
        chartRight.OnPrepareToDraw = setup;

        ActionPerSecondChart.AddChartSeries(chartLeft);
        ActionPerSecondChart.AddChartSeries(chartRight);
        ActionPerSecondChart.DrawChart();
    }

    private void ShowMostActionChart(PlayerSide side)
    {
        ChartManager chartManager = side == PlayerSide.Left ? MostActionChartLeft : MostActionChartRight;
        if (chartManager == null) return;
        
        Dictionary<string, float> mostActions = new();

        foreach (var action in side == PlayerSide.Left ? leftEventsMap : rightEventsMap)
        {
            string key = (string)action.Value.Data["Name"];

            if (mostActions.ContainsKey(key))
            {
                mostActions[key] += 1;
            }
            else
            {
                mostActions.Add(key, 0);
            }
        }

        var mostActionList = mostActions.ToList();
        mostActionList.Sort((a, b) => b.Value.CompareTo(a.Value));
        mostActionList = mostActionList.Take(3).ToList();

        ChartSeries chart = new(
            $"P1_Round_{currentRoundIndex + 1}",
            mostActionList.Select((x) => x.Value).ToArray(),
            ChartSeries.ChartType.Bar,
            side == PlayerSide.Left ? Color.green : Color.red);

        void setup()
        {
            chartManager.Setup(
                xGridSpacing: 0,
                onXLabelCreated: (index) =>
                {
                    if (index <= mostActionList.Count)
                    {
                        return mostActionList[(int)index - 1].Key;
                    }
                    else
                    {
                        return "";
                    }
                });
        }
        chart.OnPrepareToDraw = setup;

        chartManager.AddChartSeries(chart);
        chartManager.DrawChart();
    }
}

public static class ExtReplayManager
{
    public static string GetKey(this EventLog log)
    {
        if (log.Category == "Action")
        {
            return $"Action_{log.Data["Name"]}_{log.Data["Duration"] ?? null}_{log.StartedAt}";
        }
        else
        {
            return $"{log.Category}_{log.StartedAt}";
        }
    }
}