using System.Collections.Generic;
using UnityEngine;
using SumoCore;
using SumoManager;

namespace SumoBot.RuleBased.Utility
{
    public class AIBot_RuleBased_Utility : Bot
    {
        public override string ID => "Utility AI";
        public override SkillType SkillType => SkillType.Boost;

        private SumoAPI api;

        [SerializeField]
        private UtilityAI utility;

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;

            utility = new(Evaluator);
            utility.Actions.GenerateUtilityAction();
        }

        public override void OnBotUpdate()
        {
            ClearCommands();

            var scoredActions = utility.Run(2);

            foreach (var (action, _, reason) in scoredActions)
            {
                if (action is TurnAction && api.IsActionActive(action))
                    continue;

                action.Reason = reason;
                Enqueue(action);
            }

            Submit();
        }

        public override void OnBotCollision(EventParameter param)
        {
            ClearCommands();
        }

        public override void OnBattleStateChanged(BattleState state)
        {
        }

        public float Evaluator(List<ConsiderationType> types)
        {
            float result = 0;

            float angleToEnemy = api.AngleDeg();
            float distanceFromArena = api.Distance(oriPos: api.MyRobot.TransformPosition, targetPos: api.BattleInfo.ArenaPosition).magnitude;
            bool isNearArena = distanceFromArena >= (api.BattleInfo.ArenaRadius * 0.9f);
            foreach (var type in types)
            {
                switch (type)
                {
                    case ConsiderationType.DistanceFromCenter:
                        result += api.DistanceNormalized(oriPos: api.MyRobot.TransformPosition, targetPos: api.BattleInfo.ArenaPosition);
                        break;
                    case ConsiderationType.DistanceToEnemy:
                        result += api.DistanceNormalized();
                        break;
                    case ConsiderationType.EnemyInSideRight:
                        {
                            float delta = Mathf.Abs(Mathf.DeltaAngle(270, angleToEnemy));
                            result += Mathf.Clamp01((90 - delta) / 90);
                            break;
                        }
                    case ConsiderationType.EnemyInSideLeft:
                        {
                            float delta = Mathf.Abs(Mathf.DeltaAngle(90, angleToEnemy));
                            result += Mathf.Clamp01((90 - delta) / 90);
                            break;
                        }
                    case ConsiderationType.EnemyInBack:
                        {
                            float delta = Mathf.Abs(Mathf.DeltaAngle(180, angleToEnemy));
                            result += Mathf.Clamp01((45 - delta) / 45);
                            break;
                        }
                    case ConsiderationType.SkillIsReady:
                        result += api.MyRobot.Skill.IsSkillOnCooldown ? -1 : 1;
                        break;
                    case ConsiderationType.DashIsReady:
                        result += api.MyRobot.IsDashOnCooldown ? -1 : 1;
                        break;
                    case ConsiderationType.NearArena:
                        {
                            result += isNearArena ? 1 : -1;
                            break;
                        }
                    case ConsiderationType.NotNearArena:
                        {
                            result += !isNearArena ? 1 : 0.01f;
                            break;
                        }
                    case ConsiderationType.EnemyInFront:
                        {
                            float angle = api.Angle(normalized: true);
                            result += angle > 0.7f ? angle : -1;
                            break;
                        }
                    case ConsiderationType.EnemyInFrontAndClose:
                        {
                            float dist = api.DistanceNormalized();
                            float angle = api.Angle(normalized: true);
                            float val = (angle + dist) / 2;
                            result += val > 0.8f ? val : -1;
                            break;
                        }
                    default:
                        break;
                }
            }
            return result / types.Count;
        }
    }
}
