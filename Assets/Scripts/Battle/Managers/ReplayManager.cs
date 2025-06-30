using System.Collections.Generic;
using System.IO;
using System.Linq;
using SumoLog;
using UnityEditor;
using UnityEngine;
using Newtonsoft.Json;
using static SumoManager.LogManager;

public class ReplayManager : MonoBehaviour
{
    public static ReplayManager Instance { get; private set; }

    [Header("References")]
    public Transform leftBot;
    public Transform rightBot;

    [Header("Playback")]
    public float playbackSpeed = 1f;
    public bool autoStart = true;

    private List<EventLog> events = new();
    private List<TimedAction> leftActions = new();
    private List<TimedAction> rightActions = new();
    private float currentTime = 0f;
    private bool isPlaying = false;

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
        string folder = EditorUtility.OpenFolderPanel("Select Replay Folder", Path.Combine(Application.persistentDataPath, "Logs"), "");
        if (string.IsNullOrEmpty(folder)) return;

        LoadAllEvents(folder);
        PairActions();

        if (autoStart)
            isPlaying = true;
    }

    void Update()
    {
        if (!isPlaying) return;

        currentTime += Time.deltaTime * playbackSpeed;

        InterpolateBot(leftBot, leftActions);
        InterpolateBot(rightBot, rightActions);
    }

    void InterpolateBot(Transform bot, List<TimedAction> actions)
    {
        foreach (var action in actions)
        {
            float t = Mathf.InverseLerp(
                action.StartEvent.UpdatedAt,
                action.EndEvent.UpdatedAt,
                currentTime
            );

            if (t >= 0f && t <= 1f)
            {
                var start = BaseLog.FromMap(action.StartEvent.Data);
                var end = BaseLog.FromMap(action.EndEvent.Data);

                // Vector3 fromPos = new(start.Position.X, start.Robot.Position.Y, );
                // Vector3 toPos = new(end.Robot.Position.X, 0, end.Robot.Position.Y);

                // float fromRot = start.Robot.Rotation.Z;
                // float toRot = end.Robot.Rotation.Z;

                bot.position = Vector3.Lerp(start.Position, end.Position, t);
                bot.rotation = Quaternion.Lerp(Quaternion.Euler(0, 0, start.Rotation), Quaternion.Euler(0, 0, end.Rotation), t);
                break; // one active action per bot
            }
        }
    }

    void LoadAllEvents(string folderPath)
    {
        string[] files = Directory.GetFiles(folderPath, "game_*.json");
        List<EventLog> allEvents = new();

        foreach (var file in files)
        {
            string json = File.ReadAllText(file);
            var log = JsonConvert.DeserializeObject<GameLog>(json);
            foreach (var round in log.Rounds)
            {
                allEvents.AddRange(round.PlayerEvents);
            }
        }

        events = allEvents.OrderBy(e => e.UpdatedAt).ToList();
    }

    void PairActions()
    {
        var pending = new Dictionary<string, EventLog>();

        foreach (var ev in events.Where(e => e.Category == "Action"))
        {
            string key = $"{ev.Actor}_{ev.Data["Name"]}";

            if (ev.IsStart)
            {
                pending[key] = ev;
            }
            else if (pending.ContainsKey(key))
            {
                var start = pending[key];
                var range = new TimedAction { StartEvent = start, EndEvent = ev };

                if (ev.Actor == "Left")
                    leftActions.Add(range);
                else
                    rightActions.Add(range);

                pending.Remove(key);
            }
        }
    }
}

public class BotReplayState
{
    public EventLog Current;
    public EventLog Next;
}

public class TimedAction
{
    public EventLog StartEvent;
    public EventLog EndEvent;
}