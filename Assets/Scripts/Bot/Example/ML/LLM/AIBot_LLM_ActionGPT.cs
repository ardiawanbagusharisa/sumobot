using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;
using UnityEngine.Networking;

namespace ML.LanguageModels
{
    public class AIBot_LLM_ActionGPT : Bot
    {

        public override string ID => "Bot_LLM_ActionGPT";
        public override SkillType DefaultSkillType => SkillType.Stone;
        public override bool UseAsync => true;

        public string APIEndpoint = "http://localhost:9999/query";

        public bool isGenerating = false;

        private SumoAPI api;

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
        }

        public override void OnBotUpdate()
        {
            ActionRequest req = new()
            {
                state = GenerateState(),
                top_k = "1"
            };

            _ = PostInferenceAsync(req);

            Submit();
        }

        private async Task<ActionResponse> PostInferenceAsync(ActionRequest req)
        {
            if (isGenerating) return null;

            isGenerating = true;

            string json = JsonConvert.SerializeObject(req);

            using (UnityWebRequest www = new(APIEndpoint, "POST"))
            {
                byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
                www.uploadHandler = new UploadHandlerRaw(bodyRaw);
                www.downloadHandler = new DownloadHandlerBuffer();
                www.SetRequestHeader("Content-Type", "application/json");

                var operation = www.SendWebRequest();

                while (!operation.isDone)
                    await Task.Yield(); // async wait

                if (www.result == UnityWebRequest.Result.ConnectionError ||
                    www.result == UnityWebRequest.Result.ProtocolError)
                {
                    throw new Exception(www.error);
                }

                string responseText = www.downloadHandler.text;
                Logger.Info("Raw Response: " + responseText);

                isGenerating = false;
                var resp = JsonConvert.DeserializeObject<ActionResponse>(responseText);
                foreach (var action in resp.action)
                {
                    Enqueue(GetAction(action.Key, action.Value ?? 0.1f));
                }
                return resp;
            }
        }

        public string GenerateState()
        {
            var signedAngle = api.Angle();
            var signedAngleScore = api.Angle(normalized: true);
            var distanceToEnemy = 1 - api.DistanceNormalized();
            var nearArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition).magnitude / api.BattleInfo.ArenaRadius;

            var centerToMe = api.Distance(targetPos: api.MyRobot.Position, oriPos: api.BattleInfo.ArenaPosition).normalized;

            var zRot = api.MyRobot.Rotation % 360f;
            if (zRot < 0) zRot += 360f;
            Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up;

            var facingToOutside = Vector2.Dot(facingDir, centerToMe);

            // return $"GameState: BotPos=[{api.MyRobot.Position.x:F2},{api.MyRobot.Position.y:F2}], BotRot={Normalize360(api.MyRobot.Rotation):F0}, EnemyPos=[{api.EnemyRobot.Position.x:F2},{api.EnemyRobot.Position.y:F2}], EnemyRot={Normalize360(api.EnemyRobot.Rotation):F0}, EnemyAngle={signedAngle:F2}, EnemyAngleScore={signedAngleScore:F2}, EnemyDistance={distanceToEnemy:F2}, BotArena={nearArena:F2} Result:";
            return $"AngleToEnemy={signedAngle:F2}, AngleToEnemyScore={signedAngleScore:F2}, DistanceToEnemyScore={distanceToEnemy:F2}, NearBorderArenaScore={nearArena:F2}, FacingToArena={facingToOutside:F2}.";
        }

        float Normalize360(float angle)
        {
            angle %= 360f;
            if (angle < 0) angle += 360f;
            return angle;
        }

        private ISumoAction GetAction(string predictedAction, float duration)
        {
            switch (predictedAction)
            {
                case "Accelerate":
                    return new AccelerateAction(InputType.Script, Mathf.Max(0.1f, duration));
                case "TurnLeft":
                    return new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(0.1f, duration));
                case "TurnRight":
                    return new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(0.1f, duration));
                case "Dash":
                    return new DashAction(InputType.Script);
                case "Skill":
                    return new SkillAction(InputType.Script);
            }
            return new AccelerateAction(InputType.Script, 0.1f);
        }
    }

    [Serializable]
    public class ActionResponse
    {
        public Dictionary<string, float?> action;
    }

    [Serializable]
    public class ActionRequest
    {
        public string state;
        public string top_k;
    }
}