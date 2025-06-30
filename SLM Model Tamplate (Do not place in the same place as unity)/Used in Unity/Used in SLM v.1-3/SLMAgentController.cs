using System.Collections;
using UnityEngine;
using CoreSumo;
using System.Text;
using UnityEngine.Networking;
using System.Linq;
using System.Collections.Generic;

public class SLMAgentController : MonoBehaviour
{
    [Header("SLM Integration")]
    public bool EnableSLM = false;
    public string SLMApiUrl = "http://localhost:5000/predict";
    public float DecisionInterval = 0.3f;

    private SumoController controller;
    private InputProvider inputProvider;
    private SumoController enemy;
    private float timer;
    private Queue<string> pendingActions = new Queue<string>();

    void Awake()
    {
        controller = GetComponent<SumoController>();
        inputProvider = controller?.InputProvider;
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
                    Debug.Log("[SLM] Step action: " + nextAction);
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

    IEnumerator RequestStrategy()
    {
        string response = "";

        string jsonPayload = BuildSituationJson();
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

                string[] strategies = ParseStrategiesFromJson(response);

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

    string BuildSituationJson()
    {
        float enemy_distance = GetEnemyDistance();
        float enemy_angle = GetEnemyAngle();
        float edge_distance = GetEdgeDistance();
        float center_distance = GetCenterDistance();
        bool enemy_stuck = GetEnemyStuck();
        bool enemy_behind = GetEnemyBehind();
        bool skill_ready = GetSkillReady();
        bool dash_ready = GetDashReady();

        string json = "{"
            + $"\"enemy_distance\": {enemy_distance:F2},"
            + $"\"enemy_angle\": {enemy_angle:F1},"
            + $"\"edge_distance\": {edge_distance:F2},"
            + $"\"center_distance\": {center_distance:F2},"
            + $"\"enemy_stuck\": {(enemy_stuck ? 1 : 0)},"
            + $"\"enemy_behind\": {(enemy_behind ? 1 : 0)},"
            + $"\"skill_ready\": {(skill_ready ? 1 : 0)},"
            + $"\"dash_ready\": {(dash_ready ? 1 : 0)}"
            + "}";
        return json;
    }

    [SerializeField] private Vector2 arenaCenter = Vector2.zero;
    [SerializeField] private float arenaRadius = 5.0f;

    float GetEnemyDistance()
    {
        if (enemy == null) return 0f;
        return Vector2.Distance(
            new Vector2(controller.transform.position.x, controller.transform.position.y),
            new Vector2(enemy.transform.position.x, enemy.transform.position.y)
        );
    }

    float GetEnemyAngle()
    {
        if (enemy == null) return 0f;
        Vector2 dirToEnemy = (enemy.transform.position - controller.transform.position).normalized;
        Vector2 forward = controller.transform.up;
        return Vector2.SignedAngle(forward, dirToEnemy);
    }
    float GetEdgeDistance()
    {
        Vector2 pos2D = new Vector2(controller.transform.position.x, controller.transform.position.y);
        float distToCenter = Vector2.Distance(pos2D, arenaCenter);
        return Mathf.Max(0, arenaRadius - distToCenter);
    }

    float GetCenterDistance()
    {
        Vector2 pos2D = new Vector2(controller.transform.position.x, controller.transform.position.y);
        return Vector2.Distance(pos2D, arenaCenter);
    }

    bool GetEnemyStuck()
    {
        if (enemy == null) return false;
        return enemy.LastVelocity.magnitude < 0.1f;
    }

    bool GetEnemyBehind()
    {
        if (enemy == null) return false;
        Vector2 dirToEnemy = (enemy.transform.position - controller.transform.position).normalized;
        Vector2 forward = controller.transform.up;
        float angle = Vector2.Angle(forward, dirToEnemy);
        return angle > 90f;
    }

    bool GetSkillReady()
    {
        return controller != null && controller.Skill != null && !controller.Skill.IsSkillCooldown;
    }

    bool GetDashReady()
    {
        return controller != null && !controller.IsDashOnCooldown;
    }

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
}
