using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot.RuleBased.Primitive
{
    public class AIBot_Primitive : Bot
    {
        public override string ID => Name;

        public override SkillType SkillType => SkillType.Boost;

        public string Name = "Primitive";
        private const float actionInterval = 0.4f;
        private SumoAPI api;
        private BattleState currState;


        void OnPlayerBounce(PlayerSide side)
        {
            ClearCommands();
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
        }

        public override void OnBotUpdate()
        {
            float angleToEnemy = api.Angle();
            SumoBotAPI myState = api.MyRobot;
            float angleInDur = Mathf.Abs(angleToEnemy) / myState.RotateSpeed * myState.TurnRate;

            // When angle is quite enough facing the enemy, run dash, skill, accelerate action
            if (Mathf.Abs(angleToEnemy) < 20)
            {
                float distance = Vector2.Distance(api.EnemyRobot.Position, api.MyRobot.Position);

                if (!api.MyRobot.IsDashOnCooldown && distance < 2.5f)
                    Enqueue(new DashAction(InputType.Script));

                if (!api.MyRobot.Skill.IsSkillOnCooldown)
                    Enqueue(new SkillAction(InputType.Script));
            }
            else
            {
                if (angleToEnemy > 0)
                {
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, angleInDur));
                }
                else
                {
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, angleInDur));
                }

            }

            Enqueue(new AccelerateAction(InputType.Script));

            Submit();
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
            OnPlayerBounce(bounceEvent.Actor);
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
            currState = state;
            Debug.Log($"winner {winner}");
        }
    }
}