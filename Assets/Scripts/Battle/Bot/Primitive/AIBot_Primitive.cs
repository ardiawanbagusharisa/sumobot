using CoreSumo;
using UnityEngine;

namespace BotAI
{
    [CreateAssetMenu(fileName = "BOT_Primitive", menuName = "Bot/Primitive")]
    public class AIBot_Primitive : Bot
    {
        public override string ID => Name;

        public override float Interval => actionInterval;

        public string Name = "Primitive";
        private float actionInterval = 0.4f;
        private float actionTimer = 0f;
        private BotAPI api;
        private InputProvider inputProvider;
        private BattleState currState;


        void OnPlayerBounce(PlayerSide side)
        {
            inputProvider.ClearCommands();
        }

        public override void OnBotInit(PlayerSide side, BotAPI botAPI)
        {
            inputProvider = provider;
            api = botAPI;
        }

        public override void OnBotUpdate()
        {
            if (currState != BattleState.Battle_Ongoing) return;

            Enqueue(new AccelerateAction(InputType.Script));
            actionTimer -= Time.deltaTime;
            if (actionTimer <= 0f)
            {
                actionTimer = actionInterval;

                Vector2 toEnemy = (api.EnemyTransform.position - api.MyTransform.position).normalized;
                float angleDiff = Vector2.SignedAngle(api.MyTransform.up, toEnemy);

                // When angle is quite enough facing the enemy, run dash, skill, accelerate action
                if (Mathf.Abs(angleDiff) < 20)
                {
                    float distance = Vector2.Distance(api.EnemyTransform.position, api.MyTransform.position);
                    if (!api.Controller.IsDashOnCooldown && distance < 2.5f)
                    {
                        api.Controller.InputProvider.EnqueueCommand(new DashAction(InputType.Script));
                    }

                    if (!api.Controller.Skill.IsSkillCooldown)
                    {
                        api.Controller.InputProvider.EnqueueCommand(new SkillAction(InputType.Script));
                    }
                }
                else
                {
                    api.Controller.InputProvider.EnqueueCommand(new TurnAction(InputType.Script, ActionType.TurnWithAngle, angleDiff));
                }
            }

            base.OnBotUpdate();
        }

        public override void OnBotCollision(PlayerSide side)
        {
            OnPlayerBounce(side);
        }

        public override void OnBattleStateChanged(BattleState state)
        {
            currState = state;
        }
    }
}