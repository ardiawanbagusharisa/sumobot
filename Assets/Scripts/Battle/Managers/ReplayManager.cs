using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using Newtonsoft.Json;
using static SumoManager.LogManager;
using SumoLog;
using TMPro;
using System;
using System.Collections;
using UnityEngine.UI;
using SumoCore;
using SumoManager;
using System.Text;
using System.Threading.Tasks;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class ReplayManager : MonoBehaviour
{
    public static ReplayManager Instance { get; private set; }

    #region Replay Configuration properties
    [Header("Replay Configuration")]
    public bool IsEnable = false;
    public bool LoadFromPath = true;

    [Range(0f, 5f)]
    public float playbackSpeed = 1f;
    public bool autoStart = true;
    public Transform leftPlayer;
    public Transform rightPlayer;

    public CustomHandlerListener ScrollEvent;
    public CustomHandlerListener TimerSlider;
    #endregion


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
    // public TMP_Text LogUI;
    public GameObject LogItemPrefab;
    public Transform LogContainer;
    public ScrollRect LogScrollRect;
    public Scrollbar LogScrollbar;
    public TMP_Text GameDurationUI;
    public TMP_Text PlayerMode;
    public TMP_Text RoundMode;

    public TMP_Text PauseTxtUI;

    public TMP_Text LeftBotName;
    public TMP_Text LeftSkillType;
    public TMP_Text LeftWinCount;
    public TMP_Text LeftActionTaken;

    public TMP_Text RightBotName;
    public TMP_Text RightWinCount;
    public TMP_Text RightSkillType;
    public TMP_Text RightActionTaken;

    [Header("Folding Buttons")]
    public Button UnfoldDetails;
    public Button UnfoldCharts;
    public Button UnfoldLogs;
    public Button FoldDetails;
    public Button FoldCharts;
    public Button FoldLogs;
    public GameObject DetailsPanel;
    public GameObject ChartsPanel;
    public GameObject LogsPanel;
    #endregion

    #region Replay Charts
    [Header("Replay Charts")]

    public GameObject ChartContainer;
    public ChartManager Chart;
    public float EventTimeInterval = 2f;
    private static readonly string colorHexLeft = "#A5FF87C0";
    private static readonly string colorHexRight = "#FF6164C0";
    #endregion

    #region Runtime (readonly) properties 
    private readonly List<GameLog> gameLogs = new();
    private int currentGameIndex = 0;
    private int currentRoundIndex = 0;
    private float currentTime = 0f;
    public bool isPlaying = false;
    private bool isBuffer = false;

    private List<EventLog> currentRoundEvents = new();
    private float currentRoundDuration = 0f;
    private int leftEventIndex = 0;
    private int rightEventIndex = 0;
    private bool isDraggingSlider = false;
    private BattleLog metadata;
    private Dictionary<string, EventLog> leftActionMap = new();
    private Dictionary<string, EventLog> rightActionMap = new();
    private List<EventLog> leftEvents = new();
    private List<EventLog> rightEvents = new();
    private Rigidbody2D leftRigidBody;
    private Rigidbody2D rightRigidBody;
    private Dictionary<string, string> logMap = new();
    private Dictionary<string, GameObject> spawnedLogItems = new();
    private Queue<GameObject> logItemPool = new();
    private bool autoScrollLog = true;
    private Dictionary<string, bool> chartVisibilityMap = new();
    private (float, float) originalBotRotation = new();
    private (Vector2, Vector2) originalBotPosition = new();
    private bool isWaitingForRoundTransition = false;
    private float roundTransitionTimer = 0f;
    private const float ROUND_TRANSITION_DELAY = 1f; // 1 second delay before changing rounds
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

        if (GameManager.Instance.ShowReplay)
        {
            LoadGameFromBattle(Log);
            return;
        }
#if UNITY_EDITOR
        if (LoadFromPath)
            LoadGameFromPath();
#endif

    }

    void OnEnable()
    {
        if (ScrollEvent != null)
            ScrollEvent.Events[CustomHandlerListener.OnDrag].Subscribe(OnDrag);

        if (LogScrollbar != null)
        {
            LogScrollbar.onValueChanged.AddListener((val) =>
            {
                // Re-enable auto-scroll when scrolled to the bottom (where new logs appear)
                if (val < 0.05f)
                {
                    autoScrollLog = true;
                }

                Logger.Info($"LogScrollBar.onValueChanged {val}, autoScrollLog: {autoScrollLog}");
            });
        }

        if (TimeSliderUI != null)
            TimeSliderUI.onValueChanged.AddListener(OnTimeSliderChanged);

        if (TimerSlider != null)
        {
            TimerSlider.Events[CustomHandlerListener.OnPressDown].Subscribe(OnTimeSliderPointerDown);
            TimerSlider.Events[CustomHandlerListener.OnPressUp].Subscribe(OnTimeSliderPointerUp);
        }
    }


    void OnDisable()
    {
        if (TimeSliderUI != null)
            TimeSliderUI.onValueChanged.RemoveListener(OnTimeSliderChanged);

        if (PlaybackSpeedSlider != null)
            PlaybackSpeedSlider.onValueChanged.RemoveListener(OnPlayBackSpeedChanged);

        if (ScrollEvent != null)
            ScrollEvent.Events[CustomHandlerListener.OnDrag].Unsubscribe(OnDrag);

        if (TimerSlider != null)
        {
            TimerSlider.Events[CustomHandlerListener.OnPressDown].Unsubscribe(OnTimeSliderPointerDown);
            TimerSlider.Events[CustomHandlerListener.OnPressUp].Unsubscribe(OnTimeSliderPointerUp);
        }
    }


    void Update()
    {
        if (Input.GetKeyUp(KeyCode.Space))
        {
            TogglePause();
        }

        if (!isPlaying || !IsEnable || isBuffer) return;

        // Handle round transition delay
        if (isWaitingForRoundTransition)
        {
            roundTransitionTimer += Time.deltaTime;
            if (roundTransitionTimer >= ROUND_TRANSITION_DELAY)
            {
                isWaitingForRoundTransition = false;
                roundTransitionTimer = 0f;

                // Perform the actual round/game transition
                if (currentRoundIndex >= gameLogs[currentGameIndex].Rounds.Count - 1)
                {
                    if (currentGameIndex == gameLogs.Count - 1)
                    {
                        DisplayCurrentEventInfo(true);
                        isPlaying = false;
                        Logger.Info("Replay finished.");
                        return;
                    }

                    // Move to next game
                    currentGameIndex++;
                    currentRoundIndex = 0;
                }
                else
                    currentRoundIndex++;

                LoadRound(currentGameIndex, currentRoundIndex);
            }
            return;
        }

        currentTime += Time.deltaTime * playbackSpeed;

        if (TimeSliderUI != null && !isDraggingSlider)
            TimeSliderUI.value = Mathf.Ceil(currentTime);

        if (TimeLabel != null)
            TimeLabel.text = $"{FormatTime(currentTime)}/{FormatTime(currentRoundDuration)}";

        DisplayCurrentEventInfo();

        if (currentTime > currentRoundDuration)
        {
            // Start the transition delay
            isWaitingForRoundTransition = true;
            roundTransitionTimer = 0f;
        }
    }

    void FixedUpdate()
    {
        if (!isPlaying || !IsEnable || isBuffer)
            return;

        InterpolateBot(leftRigidBody, leftEvents, ref leftEventIndex);
        InterpolateBot(rightRigidBody, rightEvents, ref rightEventIndex);

    }

    void CollectEventLogs()
    {
        if (!isPlaying || !IsEnable || isBuffer)
            return;

        var lastEvents = currentRoundEvents.TakeWhile((x) => x.UpdatedAt < currentTime);

        foreach (var eventLog in lastEvents)
        {
            var key = eventLog.GetKey();

            if (eventLog.Actor == "Left")
            {
                if (eventLog.Category == "Action")
                    leftActionMap.TryAdd(eventLog.GetKey(), eventLog);
            }
            else if (eventLog.Actor == "Right")
            {
                if (eventLog.Category == "Action")
                    rightActionMap.TryAdd(eventLog.GetKey(), eventLog);
            }

            var log = eventLog.GetLogText();
            if (log != null)
                logMap.TryAdd(key, eventLog.GetLogText());
        }
    }

    private void ShowCharts()
    {
        if (ChartContainer == null || Chart == null)
            return;
        if (!ChartContainer.activeSelf)
            return;

        // With object pooling, charts render much faster
        // No need to spread across frames anymore
        ShowEventChart("Action");
        ShowEventChart("Collision");
        ShowMostActionChart();
    }
    #endregion

    void Init()
    {
        leftRigidBody = leftPlayer.gameObject.GetComponent<Rigidbody2D>();
        rightRigidBody = rightPlayer.gameObject.GetComponent<Rigidbody2D>();
        originalBotRotation = (leftRigidBody.rotation, rightRigidBody.rotation);
        originalBotPosition = (leftRigidBody.position, rightRigidBody.position);
        leftRigidBody.bodyType = RigidbodyType2D.Kinematic;
        rightRigidBody.bodyType = RigidbodyType2D.Kinematic;
        leftRigidBody.interpolation = RigidbodyInterpolation2D.Interpolate;
        rightRigidBody.interpolation = RigidbodyInterpolation2D.Interpolate;

        LoadRound(currentGameIndex, currentRoundIndex);

        if (autoStart)
        {
            isPlaying = true;
        }

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
            PlaybackSpeedLabel.text = $"Speed: {playbackSpeed:0.#}x";
    }

    void OnPlayBackSpeedChanged(float value)
    {
        playbackSpeed = value;
        if (PlaybackSpeedLabel != null)
            PlaybackSpeedLabel.text = $"Speed: {value:0.#}x";
    }

    #region Core Logics
    void AdjustLastPositionTimestamp(List<EventLog> events)
    {
        foreach (var evt in events)
        {
            if (evt.Category == "LastPosition")
            {
                evt.UpdatedAt += 0.5f;
            }
        }
    }

    void LoadRound(int gameIdx, int roundIdx)
    {
        isBuffer = true;

        leftRigidBody.MovePosition(originalBotPosition.Item1);
        leftRigidBody.MoveRotation(originalBotRotation.Item1);
        rightRigidBody.MovePosition(originalBotPosition.Item2);
        rightRigidBody.MoveRotation(originalBotRotation.Item2);

        ResetReplay(includePlayer: roundIdx == 0);

        var round = gameLogs[gameIdx].Rounds[roundIdx];

        currentRoundEvents = round.PlayerEvents.OrderBy(e => e.UpdatedAt).ToList();

        leftEvents = currentRoundEvents.Where(x => x.Actor == "Left").ToList();
        rightEvents = currentRoundEvents.Where(x => x.Actor == "Right").ToList();

        AdjustLastPositionTimestamp(leftEvents);
        AdjustLastPositionTimestamp(rightEvents);

        currentRoundDuration = currentRoundEvents.Max(e => e.UpdatedAt);

        if (TimeSliderUI != null)
        {
            TimeSliderUI.minValue = 0f;
            TimeSliderUI.maxValue = Mathf.Ceil(currentRoundDuration);
            TimeSliderUI.value = 0f;
        }

        leftEventIndex = 0;
        rightEventIndex = 0;

        isBuffer = false;

    }

    void InterpolateBot(Rigidbody2D rigidBody, List<EventLog> events, ref int index)
    {
        if (!isPlaying) return;
        if (events.Count <= 1)
            return;
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

        // Advance to the correct event pair for current time
        while (currentTime > nextEvent.UpdatedAt && index < events.Count - 1)
        {
            index++;
            currentEvent = events[index];

            // Recalculate nextEvent after index increment
            if (index == events.Count - 1)
            {
                currentEvent = events[index - 1];
                nextEvent = events[index];
                break;
            }
            else
            {
                nextEvent = events[index + 1];
            }
        }

        float t = Mathf.InverseLerp(
            currentEvent.UpdatedAt,
            nextEvent.UpdatedAt,
            currentTime
        );

        // Clamp interpolation factor to prevent overshooting
        t = Mathf.Clamp01(t);

        BaseLog start = BaseLog.FromMap(currentEvent.Data); ;
        BaseLog end = BaseLog.FromMap(nextEvent.Data); ;

        rigidBody.MovePosition(Vector3.Lerp(start.Position, end.Position, t));
        rigidBody.MoveRotation(Quaternion.Slerp(
            Quaternion.Euler(0, 0, start.Rotation),
            Quaternion.Euler(0, 0, end.Rotation),
            t
        ));
    }

#if UNITY_EDITOR
    public void LoadGameFromPath()
    {
        string basePath = Path.Combine(Application.persistentDataPath, "Logs");
        string folder = EditorUtility.OpenFolderPanel("Select Replay Folder", basePath, "");

        if (string.IsNullOrEmpty(folder)) return;

        string[] files = Directory.GetFiles(folder, "game_*.json");
        if (files.Count() == 0)
        {
            Logger.Error("Folder doesn't contain games");
            return;
        }

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
#endif

    public void LoadGameFromBattle(BattleLog battleLog)
    {
        if (battleLog.Games.Count() == 0)
        {
            Logger.Error("BattleLog doesn't contain games");
            return;
        }

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

    void DisplayCurrentEventInfo(bool isEnd = false)
    {
        // Only check for events if we have any
        if (currentRoundEvents.Count == 0)
            return;

        List<GameLog> games = gameLogs.Take(currentGameIndex + (isEnd ? 1 : 0)).ToList();
        metadata.LeftPlayerStats.WinPerGame = games.Count((i) => i.Winner == "Left");
        metadata.RightPlayerStats.WinPerGame = games.Count((i) => i.Winner == "Right");

        GameUI.SetText($"Game - {currentGameIndex + 1}");
        RoundUI.SetText($"Round - {currentRoundIndex + 1}");
        GameDurationUI?.SetText($"Duration: {metadata.BattleTime}");
        RoundMode.SetText($"Player Mode: PvP");
        RoundMode.SetText($"Round Mode: Best-of-{metadata.RoundType}");

        metadata.LeftPlayerStats.ActionTaken = leftActionMap.Count;
        metadata.RightPlayerStats.ActionTaken = rightActionMap.Count;

        LeftBotName?.SetText(metadata.LeftPlayerStats.Bot);
        LeftSkillType?.SetText(metadata.LeftPlayerStats.SkillType);
        LeftWinCount?.SetText(metadata.LeftPlayerStats.WinPerGame.ToString());
        LeftActionTaken?.SetText(metadata.LeftPlayerStats.ActionTaken.ToString());

        RightBotName?.SetText(metadata.RightPlayerStats.Bot);
        RightSkillType?.SetText(metadata.RightPlayerStats.SkillType);
        RightWinCount?.SetText(metadata.RightPlayerStats.WinPerGame.ToString());
        RightActionTaken?.SetText(metadata.RightPlayerStats.ActionTaken.ToString());

        if (!LogsPanel.activeSelf) return;

        UpdateLogUI();

        // Only auto-scroll when new logs are actually added
        if (autoScrollLog)
        {
            // Scroll to bottom since new logs are appended at the end
            LogScrollRect.verticalNormalizedPosition = 0.0f;
        }
    }

    void UpdateLogUI()
    {
        if (LogItemPrefab == null || LogContainer == null || logMap.Count == 0)
            return;

        // Spawn new log items for any logs not yet spawned
        foreach (var kvp in logMap)
        {
            if (!spawnedLogItems.ContainsKey(kvp.Key))
            {
                GameObject logItem;

                // Try to get from pool first, otherwise instantiate new
                if (logItemPool.Count > 0)
                {
                    logItem = logItemPool.Dequeue();
                    logItem.SetActive(true);
                    // Move to end when reusing from pool
                    logItem.transform.SetAsLastSibling();
                }
                else
                {
                    logItem = Instantiate(LogItemPrefab, LogContainer);
                    // New items are automatically added as last sibling
                }

                TMP_Text textComponent = logItem.GetComponent<TMP_Text>();
                if (textComponent != null)
                {
                    textComponent.text = kvp.Value;
                }
                spawnedLogItems.Add(kvp.Key, logItem);
            }
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

        // Reset transition delay when manually navigating
        isWaitingForRoundTransition = false;
        roundTransitionTimer = 0f;

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

        // Reset transition delay when manually navigating
        isWaitingForRoundTransition = false;
        roundTransitionTimer = 0f;

        LoadRound(currentGameIndex, currentRoundIndex);
        isPlaying = true;
    }

    void GoToPreviousRound()
    {
        if (currentRoundIndex > 0)
            currentRoundIndex--;

        // Reset transition delay when manually navigating
        isWaitingForRoundTransition = false;
        roundTransitionTimer = 0f;

        LoadRound(currentGameIndex, currentRoundIndex);
        isPlaying = true;
    }

    void GoToNextRound()
    {
        if (currentRoundIndex < gameLogs[currentGameIndex].Rounds.Count - 1)
            currentRoundIndex++;
        else
            currentRoundIndex = gameLogs[currentGameIndex].Rounds.Count - 1;

        // Reset transition delay when manually navigating
        isWaitingForRoundTransition = false;
        roundTransitionTimer = 0f;

        LoadRound(currentGameIndex, currentRoundIndex);
        isPlaying = true;
    }

    void OnTimeSliderChanged(float value)
    {
        if (isPlaying)
        {
            CollectEventLogs();

            if (ChartContainer != null && ChartContainer.activeSelf && isPlaying)
            {
                ShowCharts();
            }
        }

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

        leftActionMap.Clear();
        rightActionMap.Clear();
        logMap.Clear();

        // Return spawned log items to pool instead of destroying
        foreach (var logItem in spawnedLogItems.Values)
        {
            if (logItem != null)
            {
                logItem.SetActive(false);
                logItemPool.Enqueue(logItem);
            }
        }
        spawnedLogItems.Clear();

        if (Chart != null)
        {
            // Clear both chart series and side panels to prevent duplication
            Chart.ClearChartSeries();
            Chart.ClearSidePanels();
        }

        if (includePlayer)
        {
            metadata.LeftPlayerStats.ActionTaken = 0;
            metadata.RightPlayerStats.ActionTaken = 0;
        }
    }

    public void OnTimeSliderPointerDown(EventParameter _)
    {
        isDraggingSlider = true;
    }

    public void OnTimeSliderPointerUp(EventParameter _)
    {
        isDraggingSlider = false;
        isBuffer = true;

        ResetReplay(includeEvents: false, includePlayer: true);

        currentTime = TimeSliderUI.value;

        IEnumerable<EventLog> lastLeft = leftEvents.Where((x) => x.UpdatedAt < currentTime);
        IEnumerable<EventLog> lastRight = rightEvents.Where((x) => x.UpdatedAt < currentTime);

        Logger.Info($"[ReplayManager][OnTimeSliderPointerUp] Time {currentTime}, Left Events: {lastLeft.Count()}, Right Events: {lastRight.Count()}, TotalLeft Events: {leftEvents.Count()}, TotalRight Events: {rightEvents.Count()}");

        if (lastLeft.Count() > 0 || lastRight.Count() > 0)
        {
            leftEventIndex = leftEvents.IndexOf(lastLeft.LastOrDefault());
            rightEventIndex = rightEvents.IndexOf(lastRight.LastOrDefault());

            if (leftEventIndex == -1)
                leftEventIndex = 0;
            if (rightEventIndex == -1)
                rightEventIndex = 0;

            leftActionMap.Clear();
            rightActionMap.Clear();
            logMap.Clear();
        }

        InterpolateBot(leftRigidBody, leftEvents, ref leftEventIndex);
        InterpolateBot(rightRigidBody, rightEvents, ref rightEventIndex);

        StartCoroutine(ResumeAfterLoadRound());
    }

    private IEnumerator ResumeAfterLoadRound()
    {
        yield return null;

        isBuffer = false;
        isPlaying = true;
    }

    public void TogglePause()
    {
        isPlaying = !isPlaying;

        PauseTxtUI.text = isPlaying ? "Pause" : "Continue";
    }

    #endregion

    #region Chart
    string FormatTime(float time)
    {
        int minutes = Mathf.FloorToInt(time / 60f);
        int seconds = Mathf.CeilToInt(time % 60f);
        return $"{minutes:00}:{seconds:00}";
    }

    private void ShowEventChart(string category)
    {
        ChartSeries chartLeft = ChartSeries.Create(
            $"{category}/sec (Left)",
            ChartSeries.ChartType.Line, Color.green);

        ChartSeries chartRight = ChartSeries.Create(
            $"{category}/Sec (Right)",
            ChartSeries.ChartType.Line, Color.red);

        if (chartVisibilityMap.TryGetValue(chartLeft.Name, out var isLVisible))
            chartLeft.IsVisible = isLVisible;
        else
            chartVisibilityMap.Add(chartLeft.Name, chartLeft.IsVisible);

        if (chartVisibilityMap.TryGetValue(chartRight.Name, out var isRVisible))
            chartRight.IsVisible = isRVisible;
        else
            chartVisibilityMap.Add(chartRight.Name, chartRight.IsVisible);

        if (chartLeft.IsVisible || chartRight.IsVisible)
        {
            int timeFrame = 1;

            Dictionary<int, (float, float)> eventsMap = new();

            for (float i = 0; i < currentTime; i += EventTimeInterval)
            {
                int leftEventAmount = 0;
                int rightEventAmount = 0;

                if (chartLeft.IsVisible)
                {
                    Dictionary<string, EventLog> leftEventsMap = new();
                    foreach (var x in leftEvents)
                    {
                        if (category == "Collision")
                            if (x.Target.Count() == 0)
                                continue;

                        if (x.Category == category && x.UpdatedAt >= i && x.UpdatedAt < (i + EventTimeInterval))
                        {
                            leftEventsMap.TryAdd(x.GetKey(withUpdate: false), x);
                        }
                    }
                    leftEventAmount = leftEventsMap.Count();
                }


                if (chartRight.IsVisible)
                {
                    Dictionary<string, EventLog> rightEventsMap = new();
                    foreach (var x in rightEvents)
                    {
                        if (category == "Collision")
                            if (x.Target.Count() == 0)
                                continue;

                        if (x.Category == category && x.UpdatedAt >= i && x.UpdatedAt < (i + EventTimeInterval))
                        {
                            rightEventsMap.TryAdd(x.GetKey(withUpdate: false), x);
                        }
                    }
                    rightEventAmount = rightEventsMap.Count();
                }

                eventsMap[timeFrame] = (leftEventAmount, rightEventAmount);

                timeFrame += 1;
            }

            if (chartLeft.IsVisible)
                chartLeft.Data = eventsMap.Select((x) => x.Value.Item1).ToArray();

            if (chartRight.IsVisible)
                chartRight.Data = eventsMap.Select((x) => x.Value.Item2).ToArray();
        }

        chartLeft.OnVisible = (isOn) =>
        {
            chartVisibilityMap[chartLeft.Name] = isOn;
            return null;
        };
        chartRight.OnVisible = (isOn) =>
        {
            chartVisibilityMap[chartRight.Name] = isOn;
            return null;
        };

        chartLeft.OnDrawVerticalLabel = (index) =>
        {
            if (EventTimeInterval > 1.0f)
            {
                float xlabel = index * EventTimeInterval;
                return Mathf.Floor(xlabel).ToString("0.#");
            }
            return index.ToString();
        };

        chartRight.OnDrawVerticalLabel = (index) =>
        {
            if (EventTimeInterval > 1.0f)
            {
                float xlabel = index * EventTimeInterval;
                return Mathf.Floor(xlabel).ToString("0.#");
            }
            return index.ToString();
        };

        if (chartLeft.IsVisible && chartRight.IsVisible)
        {
            Chart.XGridSpacing = Mathf.FloorToInt(EventTimeInterval);
        }

        Chart.AddChartSeries(chartLeft, true);
        Chart.AddChartSeries(chartRight, true);
        Chart.DrawChart();
    }

    private void ShowMostActionChart()
    {
        var topActions = 3;

        List<string> groupLabels = new() { };
        List<Color> categoryColors = new() { };

        if (leftActionMap.Count > 0)
        {
            groupLabels.Add("Left");
            ColorUtility.TryParseHtmlString(colorHexLeft, out var colorGreen);
            categoryColors.Add(colorGreen);
        }
        if (rightActionMap.Count > 0)
        {
            groupLabels.Add("Right");
            ColorUtility.TryParseHtmlString(colorHexRight, out var colorRed);
            categoryColors.Add(colorRed);
        }

        var chart = ChartSeries.CreateGroup(
            $"Most Action Takens",
            groupNames: groupLabels.ToArray(),
            categoryColors: categoryColors.ToArray()
            );

        if (chartVisibilityMap.TryGetValue(chart.Name, out var isVisible))
            chart.IsVisible = isVisible;
        else
            chartVisibilityMap.Add(chart.Name, chart.IsVisible);

        if (chart.IsVisible)
        {
            var leftMostActs = GetChartMostAction(PlayerSide.Left, topActions);
            var rightMostActs = GetChartMostAction(PlayerSide.Right, topActions);

            List<float> data = new();
            List<string> categories = new();

            for (int i = 0; i < leftMostActs.Count(); i++)
            {
                data.Add(leftMostActs[i].Value);
                categories.Add(leftMostActs[i].Key);
            }
            for (int i = 0; i < rightMostActs.Count(); i++)
            {
                data.Add(rightMostActs[i].Value);
                categories.Add(rightMostActs[i].Key);
            }

            chart.Data = data.ToArray();
            chart.CategoryNames = categories.ToArray();
        }

        chart.OnVisible = (isOn) =>
        {
            chartVisibilityMap[chart.Name] = isOn;
            return null;
        };

        Chart.AddChartSeries(chart, true);
        Chart.DrawChart();

    }

    private List<KeyValuePair<string, float>> GetChartMostAction(PlayerSide side, int topActions)
    {
        Dictionary<string, float> mostActions = new();

        foreach (var action in side == PlayerSide.Left ? leftActionMap : rightActionMap)
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
        mostActionList = mostActionList.Take(topActions).ToList();
        return mostActionList;
    }

    public void ShowChart()
    {
        if (ChartContainer == null) return;
        ChartContainer.SetActive(true);
        if (!isPlaying)
        {
            ShowCharts();
        }
    }
    public void HideChart()
    {
        ChartContainer.SetActive(false);
    }
    #endregion

    public void OnDrag(EventParameter _)
    {
        Logger.Info("OnDrag");
        autoScrollLog = false;
    }

    public void BackToBattle()
    {
        SFXManager.Instance?.Play2D("ui_accept");
        GameManager.Instance?.Replay_BackToBattle();
    }

    public void ToggleDetails()
    {
        if (DetailsPanel != null)
        {
            SFXManager.Instance?.Play2D("ui_accept");
            bool isActive = DetailsPanel.activeSelf;
            DetailsPanel.SetActive(!isActive);
            FoldDetails.gameObject.SetActive(!isActive);
            UnfoldDetails.gameObject.SetActive(isActive);
        }
    }

    public void ToggleCharts()
    {
        if (ChartsPanel != null)
        {
            SFXManager.Instance?.Play2D("ui_accept");
            bool isActive = ChartsPanel.activeSelf;
            ChartsPanel.SetActive(!isActive);
            FoldCharts.gameObject.SetActive(!isActive);
            UnfoldCharts.gameObject.SetActive(isActive);
        }
    }

    public void ToggleLogs()
    {
        if (LogsPanel != null)
        {
            SFXManager.Instance?.Play2D("ui_accept");
            bool isActive = LogsPanel.activeSelf;
            LogsPanel.SetActive(!isActive);
            FoldLogs.gameObject.SetActive(!isActive);
            UnfoldLogs.gameObject.SetActive(isActive);
        }
    }
}

public static class ExtReplayManager
{
    private static readonly string colorHexLeft = "#A5FF87C0";
    private static readonly string colorHexRight = "#FF6164C0";

    public static string GetKey(this EventLog log, bool withUpdate = true)
    {
        if (log.Category == "Action")
        {
            string result = $"{log.Actor}_Action_{log.Data["Name"]}_{log.Data["Duration"] ?? null}_{log.StartedAt}";
            if (withUpdate)
            {
                result += $"_{log.UpdatedAt}";
            }
            return result;
        }
        else
        {
            return $"{log.Actor}_{log.Category}_{log.StartedAt}";
        }
    }

    public static string GetLogText(this EventLog log)
    {
        var color = log.Actor == "Left" ? colorHexLeft : colorHexRight;

        if (log.Category == "Action")
        {
            return $"<color={color}>[{log.StartedAt:F2}] {log.Actor} | Action | {log.Data["Name"]} | {log.Data["Duration"]:F2}</color>";
        }
        else if (log.Category == "Collision")
        {
            return $"[{log.StartedAt:F2}] {log.Actor} | Collision | Impact={log.Data["Impact"]:F2} Dur={log.Data["Duration"]:F2} Lock={log.Data["LockDuration"]:F2}";
        }
        return null;
    }
}

