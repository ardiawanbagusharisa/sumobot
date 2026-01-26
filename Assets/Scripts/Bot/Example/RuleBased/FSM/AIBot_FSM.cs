using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public enum BotState
    {
        Searching,
        Approaching,
        Attacking,
        Dodging,
        Recovering,
        Idle
    }

    public class AlBot_FSM : Bot
    {
        #region Runtime Properties
        public override string ID => "Bot_FSM";
        public override SkillType DefaultSkillType => SkillType.Boost;
        #endregion

        #region State Threshold Properties
        [Header("State Transition Thresholds")]
        public float approachAngle = 20f;
        public float attackAngle = 15f;
        public float searchAngle = 45f;
        public float approachDistance = 5.0f;
        public float attackDistance = 2.5f;
        public float searchRatio = 0.7f;
        #endregion

        #region Behaviour Thresholds Properties
        [Header("Action & Behavior Modifiers")]
        public float focusAngle = 3f;
        public float dashSkillAngle = 10f;
        public float searchTurnAngle = 15f;
        public float minTurnDuration = 0.1f;
        public float dodgeDuration = 0.2f;
        public float recoveryDuration = 0.1f;
        public float minAccelerationRatio = 0.9f;        
        public float stableLinearVelocity = 1.0f;
        public float stableAngularVelocity = 10.0f;
        public float edgeAvoidDistance = 1f;
        #endregion

        #region Runtime Properties
        private SumoAPI api;
        private PlayerSide mySide;
        private BotState currentState = BotState.Searching;
        private float stateTimer = 0f;
        #endregion

        #region Overload Methods
        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            currentState = BotState.Searching;
            stateTimer = 0f;
        }

        public override void OnBotUpdate()
        {
            ClearCommands();
            SumoBotAPI myRobot = api.MyRobot;
            SumoBotAPI enemyRobot = api.EnemyRobot;
            stateTimer += Time.deltaTime; 

            switch (currentState)
            {
                case BotState.Searching:
                    HandleSearching(myRobot, enemyRobot);
                    break;
                case BotState.Approaching:
                    HandleApproaching(myRobot, enemyRobot);
                    break;
                case BotState.Attacking:
                    HandleAttacking(myRobot, enemyRobot);
                    break;
                case BotState.Dodging:
                    HandleDodging(myRobot, enemyRobot);
                    break;
                case BotState.Recovering:
                    HandleRecovering(myRobot, enemyRobot);
                    break;
                case BotState.Idle:
                    HandleIdle(myRobot, enemyRobot);
                    break;
            }
            Submit(); 
            Logger.Info($"Current State: {currentState}");
        }

        public override void OnBotCollision(BounceEvent param)
        {
            if (param.Actor != mySide)
            {
                ClearCommands();
                BotState newState = currentState == BotState.Attacking ? BotState.Attacking : BotState.Dodging;
                TransitionToState(newState);
            }
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
            if (state == BattleState.Battle_Ongoing)
                TransitionToState(BotState.Searching);
            else if (state == BattleState.Battle_End)
            {
                TransitionToState(BotState.Idle);
                ClearCommands();
            }
        }
        #endregion

        #region State Handlings
        private void TransitionToState(BotState newState)
        {
            if (currentState == newState) return;
            Logger.Info($"Transitioning from {currentState} to {newState}");
            currentState = newState;
            stateTimer = 0f;
            ClearCommands();
        }

        private void HandleSearching(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
        {
            float angleToEnemy = api.Angle();
            float distanceToEnemy = Vector3.Distance(enemyRobot.Position, myRobot.Position);

            if (Mathf.Abs(angleToEnemy) < approachAngle && distanceToEnemy < approachDistance)
            {
                TransitionToState(BotState.Approaching);
                return;
            }

            if (myRobot.LinearVelocity.magnitude < myRobot.MoveSpeed * minAccelerationRatio)
            {
                if (Vector3.Distance(myRobot.Position, api.BattleInfo.ArenaPosition) > api.BattleInfo.ArenaRadius * searchRatio || distanceToEnemy > approachDistance)
                    Enqueue(new AccelerateAction(InputType.Script));
            }

            if (Mathf.Abs(angleToEnemy) > focusAngle)
            {
                float turnAmount = Mathf.Min(Mathf.Abs(angleToEnemy), searchTurnAngle);
                turnAmount = Mathf.Max(turnAmount, minTurnDuration * myRobot.RotateSpeed);

                if (angleToEnemy > 0)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, turnAmount));
                else
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, turnAmount));
            }
        }

        private void HandleApproaching(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
        {
            float angleToEnemy = api.Angle();
            float distanceToEnemy = Vector3.Distance(enemyRobot.Position, myRobot.Position);

            if (distanceToEnemy < attackDistance && Mathf.Abs(angleToEnemy) < attackAngle)
            {
                TransitionToState(BotState.Attacking);
                return;
            }
            else if (Mathf.Abs(angleToEnemy) > searchAngle)
            {
                TransitionToState(BotState.Searching);
                return;
            }
            else if (distanceToEnemy > api.BattleInfo.ArenaRadius + 1f)
            {
                TransitionToState(BotState.Searching);
                return;
            }

            if (Mathf.Abs(angleToEnemy) > focusAngle)
            {
                float angleInDur = Mathf.Abs(angleToEnemy) / myRobot.RotateSpeed;
                angleInDur = Mathf.Max(angleInDur, minTurnDuration);

                if (angleToEnemy > 0)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, angleInDur));
                else
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, angleInDur));
            }

            if (myRobot.LinearVelocity.magnitude < myRobot.MoveSpeed * minAccelerationRatio)
                Enqueue(new AccelerateAction(InputType.Script));
        }

        private void HandleAttacking(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
        {
            float angleToEnemy = api.Angle();
            float distanceToEnemy = Vector3.Distance(enemyRobot.Position, myRobot.Position);

            if (distanceToEnemy > approachDistance || Mathf.Abs(angleToEnemy) > approachAngle)
            {
                TransitionToState(BotState.Approaching);
                return;
            }

            if (Mathf.Abs(angleToEnemy) > focusAngle)
            {
                float angleInDur = Mathf.Abs(angleToEnemy) / myRobot.RotateSpeed;
                angleInDur = Mathf.Max(angleInDur, minTurnDuration);

                if (angleToEnemy > 0)
                {
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, angleInDur));
                }
                else
                {
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, angleInDur));
                }
            }

            if (!myRobot.IsDashOnCooldown && Mathf.Abs(angleToEnemy) < dashSkillAngle)
            {
                Enqueue(new DashAction(InputType.Script));
            }

            // Skill activation regardless of type (Boost or Stone)
            if (!myRobot.Skill.IsSkillOnCooldown && Mathf.Abs(angleToEnemy) < dashSkillAngle)
            {
                Enqueue(new SkillAction(InputType.Script));
            }

            if (myRobot.LinearVelocity.magnitude < myRobot.MoveSpeed * minAccelerationRatio)
            {
                Enqueue(new AccelerateAction(InputType.Script));
            }
        }

        private void HandleDodging(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
        {
            Vector2 directionToEnemy = (enemyRobot.Position - myRobot.Position).normalized;
            Vector2 dodgeDirection = new Vector2(-directionToEnemy.y, directionToEnemy.x); 
            Vector2 arenaCenter = api.BattleInfo.ArenaPosition;

            if (Vector2.Distance(myRobot.Position, arenaCenter) > api.BattleInfo.ArenaRadius - edgeAvoidDistance)
                dodgeDirection = (arenaCenter - myRobot.Position).normalized;
            else
            {
                // Randomize dodge direction slightly
                float randomAngle = Random.Range(-Mathf.PI / 4, Mathf.PI / 4); // Random angle between -45 and 45 degrees
                dodgeDirection = Quaternion.Euler(0, 0, randomAngle) * dodgeDirection;
            }

            float dodgeTurnDuration = Random.Range(dodgeDuration / 2f, dodgeDuration * 2f);
            Enqueue(new TurnAction(InputType.Script, dodgeDirection.x > 0 ? ActionType.TurnLeft : ActionType.TurnRight, dodgeTurnDuration));

            if (myRobot.LinearVelocity.magnitude < myRobot.MoveSpeed * minAccelerationRatio)
                Enqueue(new AccelerateAction(InputType.Script));

            if (stateTimer >= dodgeDuration)
                TransitionToState(BotState.Recovering);
        }

        private void HandleRecovering(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
        {
            if (stateTimer >= recoveryDuration || (myRobot.LinearVelocity.magnitude < stableLinearVelocity && Mathf.Abs(myRobot.AngularVelocity) < stableAngularVelocity))
                TransitionToState(BotState.Searching);

            if (myRobot.LinearVelocity.magnitude < myRobot.MoveSpeed * minAccelerationRatio)
            {
                Vector2 directionToCenter = (api.BattleInfo.ArenaPosition - myRobot.Position).normalized;
                Vector2 forward = Quaternion.Euler(0, 0, myRobot.Rotation) * Vector2.up;
                float angleToCenter = Vector2.SignedAngle(forward, directionToCenter);
                
                if (angleToCenter < 0)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, Mathf.Max(minTurnDuration, Mathf.Abs(angleToCenter) / myRobot.RotateSpeed)));
                else if (angleToCenter > 0)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, Mathf.Max(minTurnDuration, Mathf.Abs(angleToCenter) / myRobot.RotateSpeed)));
                
                float distance = Vector2.Distance(myRobot.Position, api.BattleInfo.ArenaPosition);
                if (distance < api.BattleInfo.ArenaRadius - edgeAvoidDistance)
                    Enqueue(new AccelerateAction(InputType.Script));
            }
        }

        private void HandleIdle(SumoBotAPI myRobot, SumoBotAPI enemyRobot)
        {
            if (api.BattleInfo.CurrentState == BattleState.Battle_Ongoing)
                TransitionToState(BotState.Searching);
        }
        #endregion
    }
}