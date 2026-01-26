
using System.Linq;
using SumoCore;
using SumoInput;
using SumoLog;
using SumoManager;
using Unity.InferenceEngine;
using UnityEngine;

namespace SumoBot
{
    public class AIBot_Template : Bot
    {
        public override string ID => "Bot_Template";
        public override SkillType DefaultSkillType => SkillType.Boost;

        private SumoAPI api;

        // Where the battle state changes
        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
        }

        // When your AI is got a collision (bounce)
        public override void OnBotCollision(BounceEvent bounceEvent)
        {
            PlayerSide hitter = bounceEvent.Actor;

            if (hitter == api.MyRobot.Side)
                Logger.Info($"My AI sent hit to enemy!");
            else
                Logger.Info($"My AI got hit!");

        }

        // Initial state of your script
        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
        }

        // OnBotUpdate() will be called everytime when the [ElapsedTime] of Battle => [Interval]
        public override void OnBotUpdate()
        {
            // If the robot is facing the enemy 90%, queue accelerate
            if (api.Angle(normalized: true) > 0.9)
            {
                Enqueue(new AccelerateAction(InputType.Script));
            }

            // Example 
            float distanceScore = api.DistanceNormalized();
            float angleScore = api.Angle(normalized: true);
            float angleToEnemy = api.Angle();
            float angleToArena = api.Angle(targetPos: api.BattleInfo.ArenaPosition);
            float distanceFromArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition).magnitude / api.BattleInfo.ArenaRadius;
            var faceTowardArena = FacingFormula(targetPos: api.BattleInfo.ArenaPosition);
            var faceTowardEnemy = FacingFormula(targetPos: api.EnemyRobot.Position);

            Logger.Info($"[{api.MyRobot.Side}] Template.OnBotUpdate:\nDistance Score: {distanceScore}\nAngle Score: {angleScore}\n AngleToEnemy: {angleToEnemy}\n Distance From Arena: {distanceFromArena}\n Face Toward Arena: {faceTowardArena}\n Face Toward Enemy: {faceTowardEnemy}");

            var previousEvent = api.Log.GetLastEvents();

            var list = previousEvent.Select((x) =>
            {
                var log = x.RobotLog;
                if (log is ActionLog action)
                {
                    return $"Last action: {action.Action.Name} at {x.StartedAt}";
                }
                if (log is CollisionLog collision)
                {
                    return $"Last collision {collision.Impact} at {x.StartedAt}";
                }
                return "Undefined type";
            });

            Logger.Info($"Last Actions {previousEvent.Count}: {string.Join("\n", list)}");


            // To activate the queued actions
            // Submit();
        }

        // Ranging from -1 to 1
        // -1 towards the 
        float FacingFormula(
            Vector2? oriPos = null,
            float? oriRot = null,
            Vector2? targetPos = null)
        {
            var dist = api.Distance(targetPos: oriPos ?? api.MyRobot.Position, oriPos: targetPos ?? api.BattleInfo.ArenaPosition).normalized;
            var zRot = oriRot ?? api.MyRobot.Rotation % 360f;
            if (zRot < 0) zRot += 360f;
            Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up;
            return Vector2.Dot(facingDir, dist);
        }
    }

}