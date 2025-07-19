using UnityEngine;
using System.Text;
using UnityEngine.Networking;
using System.Linq;
using System.Collections.Generic;
using SumoCore;
using SumoManager;
using SumoBot;
using System.Threading.Tasks;

public class SLMAgentController : Bot
{
    [Header("SLM Integration")]
    public bool EnableSLM = false;
    public string SLMApiUrl = "http://localhost:5000/predict";
    public float DecisionInterval = 0.1f;
    public int contextWindow = 3;
    private Queue<string> contextBuffer;
    private float timer;
    private Queue<string> pendingActions;
    private SumoAPI api;
    private BattleState currentState;

    public override string ID => "SLM";

    public override float Interval => DecisionInterval;

    public override SkillType SkillType => SkillType.Boost;

    #region Bot methods
    public override void OnBotInit(PlayerSide side, SumoAPI botAPI)
    {
        api = botAPI;
        contextBuffer = new Queue<string>();
        pendingActions = new Queue<string>();
        timer = 0f;
    }

    public override void OnBotCollision(ActionParameter param)
    {

    }

    public override void OnBattleStateChanged(BattleState state)
    {
        currentState = state;
    }

    public override void OnBotUpdate()
    {
        RunSLM();
        base.OnBotUpdate();
    }
    #endregion

    #region SLM methods
    void RunSLM()
    {
        if (!EnableSLM) return;


        if (currentState != BattleState.Battle_Ongoing)
            return;

        timer += Time.deltaTime;
        if (timer >= DecisionInterval)
        {
            timer = 0f;

            if (pendingActions.Count > 0)
            {
                ClearCommands();
                while (pendingActions.Count > 0)
                {
                    string nextAction = pendingActions.Dequeue();
                    ISumoAction sumoAction = StrategyToActionMapper.Map(nextAction, api);
                    if (sumoAction != null)
                        Enqueue(sumoAction);
                    else
                        Debug.LogWarning("SLMAgentController: Unknown strategy: " + nextAction);
                }
            }
            else
            {
                _ = RequestStrategyAsync();
            }
        }
    }

    async Task RequestStrategyAsync()
    {
        string situationStr = BuildSituationString();
        contextBuffer.Enqueue(situationStr);
        if (contextBuffer.Count > contextWindow)
            contextBuffer.Dequeue();
        string contextInput = string.Join(" [CTX] ", contextBuffer.ToArray());
        string jsonPayload = "{\"context_input\": \"" + contextInput + "\"}";

        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonPayload);
        using UnityWebRequest req = new UnityWebRequest(SLMApiUrl, "POST");
        req.uploadHandler = new UploadHandlerRaw(bodyRaw);
        req.downloadHandler = new DownloadHandlerBuffer();
        req.SetRequestHeader("Content-Type", "application/json");

        var operation = req.SendWebRequest();

        while (!operation.isDone)
            await Task.Yield(); // wait for completion

        if (req.result == UnityWebRequest.Result.Success)
        {
            string response = req.downloadHandler.text;
            string[] strategies = ParseStrategiesFromJson(response);

            var actions = strategies
                .SelectMany(strat => strat.Contains("+")
                    ? strat.Split('+').Select(s => s.Trim().ToLower())
                    : new string[] { strat.Trim().ToLower() })
                .ToList();

            // Sort by priority
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
            Debug.LogWarning("SLM request failed: " + req.error + " " + req.downloadHandler.text);
        }

        Debug.Log("[SLM] Payload: " + jsonPayload);
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

    void LogSLMRequest(string jsonPayload)
    {
        Debug.Log("[SLM] Request payload: " + jsonPayload);
    }

    void LogSLMResponse(string response)
    {
        Debug.Log("[SLM] Response from API: " + response);
    }
    #endregion

    #region Robot utility methods
    float GetEnemyDistance()
    {
        return Vector3.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
    }

    float GetEnemyAngle()
    {
        Vector3 toEnemy = (api.EnemyRobot.Position - api.MyRobot.Position).normalized;
        float angle = Vector3.SignedAngle(api.MyRobot.Rotation * Vector3.forward, toEnemy, Vector3.up);
        return angle;
    }

    float GetEdgeDistance()
    {
        float arenaRadius = api.BattleInfo.ArenaRadius;
        float distanceFromCenter = new Vector2(api.MyRobot.Position.x, api.MyRobot.Position.z).magnitude;
        return Mathf.Max(0f, arenaRadius - distanceFromCenter);
    }

    float GetCenterDistance()
    {
        return new Vector2(api.MyRobot.Position.x, api.MyRobot.Position.z).magnitude;
    }

    bool GetEnemyStuck()
    {
        float minSpeed = 0.05f;
        return api.EnemyRobot.LinearVelocity.magnitude < minSpeed;
    }

    bool GetEnemyBehind()
    {
        Vector2 toEnemy = (api.EnemyRobot.Position - api.MyRobot.Position).normalized;
        float angle = Vector2.Angle(api.MyRobot.Rotation * Vector3.up, toEnemy);
        return angle > 120f;
    }

    bool GetSkillReady()
    {
        return !api.MyRobot.Skill.IsSkillOnCooldown;
    }

    bool GetDashReady()
    {
        return !api.MyRobot.IsDashOnCooldown;
    }
    #endregion
}