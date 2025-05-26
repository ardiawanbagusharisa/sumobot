using BattleLoop;
using CoreSumoRobot;
using UnityEngine;

namespace BotAI
{
    public class AIBot_Primitive : MonoBehaviour
    {
        private SumoRobotController controller;
        private SumoRobotController enemy;
        private float actionInterval = 0.4f;
        private float actionTimer = 0f;
        void OnEnable()
        {
            controller = GetComponent<SumoRobotController>();
            controller.OnPlayerBounce += OnPlayerBounce;
        }

        void FixedUpdate()
        {
            if (enemy == null)
            {
                if (controller.Side == PlayerSide.Left)
                {
                    enemy = BattleManager.Instance.Battle.RightPlayer;
                }
                else
                {
                    enemy = BattleManager.Instance.Battle.LeftPlayer;
                }
            }
        }

        void Update()
        {
            if (enemy == null) return;
            if (BattleManager.Instance.CurrentState != BattleState.Battle_Ongoing) return;
            controller.InputProvider.EnqueueCommand(new AccelerateAction(InputType.Script));
            actionTimer -= Time.deltaTime;
            if (actionTimer <= 0f)
            {
                actionTimer = actionInterval;

                Vector2 toEnemy = (enemy.transform.position - transform.position).normalized;
                float angleDiff = Vector2.SignedAngle(transform.up, toEnemy);

                // When angle is quite enough facing the enemy, run dash, skill, accelerate action
                if (Mathf.Abs(angleDiff) < 20)
                {
                    float distance = Vector2.Distance(enemy.transform.position, transform.position);
                    if (!controller.IsDashCooldown && distance < 2.5f)
                    {
                        controller.InputProvider.EnqueueCommand(new DashAction(InputType.Script));
                    }

                    if (!controller.Skill.IsSkillCooldown)
                    {
                        controller.InputProvider.EnqueueCommand(new SkillAction(InputType.Script));
                    }
                }
                else
                {
                    controller.InputProvider.EnqueueCommand(new TurnAngleAction(angleDiff));
                }
            }
        }

        void OnPlayerBounce(PlayerSide side)
        {
            controller.InputProvider.ClearCommands();
        }
    }
}