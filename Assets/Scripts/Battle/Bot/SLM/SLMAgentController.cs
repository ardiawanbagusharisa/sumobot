using System.Collections;
using UnityEngine;
using System.Text;
using UnityEngine.Networking;
using System.Linq;
using System.Collections.Generic;
using SumoInput;
using SumoCore;
using SumoManager;

public class SLMAgentController : MonoBehaviour
{
    [Header("SLM Integration")]
    public bool EnableSLM = false;
    public string SLMApiUrl = "http://localhost:5000/predict";
    public float DecisionInterval = 0.3f;
    public int contextWindow = 3;
    private Queue<string> contextBuffer;
    private SumoController controller;
    private InputProvider inputProvider;
    private SumoController enemy;
    private float timer;
    private Queue<string> pendingActions;

    // === Unity Lifecycle Methods ===
    void Awake()
    {
        controller = GetComponent<SumoController>();
        inputProvider = controller?.InputProvider;
        contextBuffer = new Queue<string>();
        pendingActions = new Queue<string>();
        timer = 0f;
    }

    void Start()
    {
        TryAssignEnemy();
    }
    void Update()
    {
        if (!EnableSLM) return;

        if (controller == null || inputProvider == null)
        {
            controller = GetComponent<SumoController>();
            inputProvider = controller?.InputProvider;
            if (controller == null || inputProvider == null) return;
        }

        if (enemy == null)
        {
            TryAssignEnemy();
            if (enemy == null) return;
        }

        if (BattleManager.Instance.CurrentState != BattleState.Battle_Ongoing)
            return;

        timer += Time.deltaTime;
        if (timer >= DecisionInterval)
        {
            timer = 0f;

            if (pendingActions.Count > 0)
            {
                inputProvider.ClearCommands();
                while (pendingActions.Count > 0)
                {
                    string nextAction = pendingActions.Dequeue();
                    ISumoAction sumoAction = StrategyToActionMapper.Map(nextAction);
                    if (sumoAction != null)
                        inputProvider.EnqueueCommand(sumoAction);
                    else
                        Debug.LogWarning("SLMAgentController: Unknown strategy: " + nextAction);
                }
            }
            else
            {
                StartCoroutine(RequestStrategy());
            }
        }
    }

    // === SLM & Enemy Setup ===
    void TryAssignEnemy()
    {
        var bm = BattleManager.Instance;
        if (bm != null && bm.Battle != null)
        {
            enemy = controller.Side == PlayerSide.Left
                ? bm.Battle.RightPlayer
                : bm.Battle.LeftPlayer;
        }
    }

    // === SLM Strategy Request Logic ===
    IEnumerator RequestStrategy()
    {
        string response = "";

        // === Build context window: concatenation of last N instructions ===
        string situationStr = BuildSituationString();
        contextBuffer.Enqueue(situationStr);
        if (contextBuffer.Count > contextWindow)
            contextBuffer.Dequeue();
        string contextInput = string.Join(" [CTX] ", contextBuffer.ToArray());
        string jsonPayload = "{\"context_input\": \"" + contextInput + "\"}";

        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);

        using (UnityWebRequest req = new UnityWebRequest(SLMApiUrl, "POST"))
        {
            req.uploadHandler = new UploadHandlerRaw(bodyRaw);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            yield return req.SendWebRequest();

            if (req.result == UnityWebRequest.Result.Success)
            {
                response = req.downloadHandler.text;

                // === Parsing response strategy ===
                string[] strategies = ParseStrategiesFromJson(response);

                // Split combo actions, order priority (turn > dash > accelerate > boost)
                var actions = strategies
                    .SelectMany(strat => strat.Contains("+")
                        ? strat.Split('+').Select(s => s.Trim().ToLower())
                        : new string[] { strat.Trim().ToLower() })
                    .ToList();

                actions.Sort((a, b) =>
                {
                    if (a.StartsWith("turn") && !b.StartsWith("turn")) return -1;
                    if (!a.StartsWith("turn") && b.StartsWith("turn")) return 1;
                    if (a.StartsWith("dash") && !b.StartsWith("dash")) return -1;
                    if (!a.StartsWith("dash") && b.StartsWith("dash")) return 1;
                    if (a.StartsWith("accelerate") && !b.StartsWith("accelerate")) return -1;
                    if (!a.StartsWith("accelerate") && b.StartsWith("accelerate")) return 1;
                    return 0;
                });

                pendingActions = new Queue<string>(actions);

                Debug.Log("[SLM] Queued actions: " + string.Join(", ", actions));
            }
            else
            {
                Debug.LogWarning("SLMAgentController: Request failed: " + req.error + " " + req.downloadHandler.text);
            }

            Debug.Log("[SLM] Payload: " + jsonPayload);
            Debug.Log("[SLM] Response: " + response);
        }
    }

    string BuildSituationString()
    {
        float enemy_distance = GetEnemyDistance();
        float enemy_angle = GetEnemyAngle();
        float edge_distance = GetEdgeDistance();
        float center_distance = GetCenterDistance();
        bool enemy_stuck = GetEnemyStuck();
        bool enemy_behind = GetEnemyBehind();
        bool skill_ready = GetSkillReady();
        bool dash_ready = GetDashReady();

        return $"Enemy is {enemy_distance:F2} meters ahead, angle {enemy_angle:F1} degrees, edge distance {edge_distance:F2}, center distance {center_distance:F2}, enemy stuck: {(enemy_stuck ? 1 : 0)}, enemy behind: {(enemy_behind ? 1 : 0)}, skill ready: {(skill_ready ? 1 : 0)}, dash ready: {(dash_ready ? 1 : 0)}";
    }

    // === Buffer Context Window Logic ===
    void UpdateBuffer(string currentSituation)
    {
        contextBuffer.Enqueue(currentSituation);
        if (contextBuffer.Count > contextWindow)
            contextBuffer.Dequeue();
    }

    // === Parsing & Action Mapping ===
    string[] ParseStrategiesFromJson(string json)
    {
        if (json.Contains("[") && json.Contains("]"))
        {
            int start = json.IndexOf("[") + 1;
            int end = json.IndexOf("]", start);
            string inside = json.Substring(start, end - start).Replace("\"", "").Trim();
            string[] arr = inside.Split(new char[] { ',' }, System.StringSplitOptions.RemoveEmptyEntries);
            for (int i = 0; i < arr.Length; i++)
                arr[i] = arr[i].Trim();
            return arr;
        }
        else if (json.Contains(":") && json.Contains("\""))
        {
            int idx = json.IndexOf(":");
            int idx2 = json.LastIndexOf("\"");
            if (idx != -1 && idx2 != -1)
            {
                return new string[] { json.Substring(idx + 2, idx2 - idx - 2) };
            }
        }
        return new string[0];
    }

    // === Game State Feature Extraction ===
    float GetEnemyDistance()
    {
        if (controller == null || enemy == null) return 0f;
        return Vector3.Distance(controller.transform.position, enemy.transform.position);
    }

    float GetEnemyAngle()
    {
        if (controller == null || enemy == null) return 0f;
        Vector3 toEnemy = (enemy.transform.position - controller.transform.position).normalized;
        float angle = Vector3.SignedAngle(controller.transform.forward, toEnemy, Vector3.up);
        return angle;
    }

    float GetEdgeDistance()
    {
        if (controller == null) return 0f;
        float arenaRadius = 5f;
        float distanceFromCenter = new Vector2(controller.transform.position.x, controller.transform.position.z).magnitude;
        return Mathf.Max(0f, arenaRadius - distanceFromCenter);
    }

    float GetCenterDistance()
    {
        if (controller == null) return 0f;
        return new Vector2(controller.transform.position.x, controller.transform.position.z).magnitude;
    }

    bool GetEnemyStuck()
    {
        if (enemy == null) return false;
        float minSpeed = 0.05f;
        return enemy.LastVelocity.magnitude < minSpeed;
    }

    bool GetEnemyBehind()
    {
        if (controller == null || enemy == null) return false;
        Vector2 toEnemy = (enemy.transform.position - controller.transform.position).normalized;
        float angle = Vector2.Angle(controller.transform.up, toEnemy);
        return angle > 120f;
    }

    bool GetSkillReady()
    {
        return controller != null && controller.isSkillReady;
    }

    bool GetDashReady()
    {
        return controller != null && controller.isDashReady;
    }

    // === Action Execution ===
    void EnqueueAction(string action)
    {
        ISumoAction sumoAction = StrategyToActionMapper.Map(action);
        if (sumoAction != null)
        {
            inputProvider.EnqueueCommand(sumoAction);
            Debug.Log("[SLM] Action enqueued: " + action);
        }
        else
        {
            Debug.LogWarning("[SLM] Unknown strategy action: " + action);
        }
    }

    // === Debugging & Logging ===
    void LogSLMRequest(string jsonPayload)
    {
        Debug.Log("[SLM] Request payload: " + jsonPayload);
    }

    void LogSLMResponse(string response)
    {
        Debug.Log("[SLM] Response from API: " + response);
    }
}