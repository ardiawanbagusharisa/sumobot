using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public enum UltimateState { Hunt, Attack, Ranbu, Recover }

    public class AIBot_FSM_Anandan : Bot
    {
        public override string ID => "FSM_Anandan";
        public override SkillType DefaultSkillType => SkillType.Boost;

        private const float FOCUS_ANGLE = 4f;
        private const float ATTACK_ANGLE = 25f;
        private const float EXIT_ANGLE = 55f;
        private const float DASH_ANGLE = 30f;
        private const float SKILL_ANGLE = 35f;

        private const float ATTACK_DIST = 5.0f;
        private const float EXIT_DIST = 9.0f;

        private const float EDGE_CAUTION = 0.62f;
        private const float EDGE_ENTER = 0.78f;
        private const float EDGE_EXIT = 0.50f;
        private const float EDGE_DESPERATION = 0.91f;

        private const float ENEMY_EDGE_RANBU = 0.72f;
        private const float ENEMY_EDGE_EXIT = 0.58f;
        private const float RANBU_DASH_ANGLE = 45f;
        private const float RANBU_SKILL_ANGLE = 55f;

        private const float PREDICT_TIME = 0.15f;
        private const float POSITION_DOT_THRESHOLD = -0.3f;
        private const float DASH_MOMENTUM_RATIO = 0.4f;

        private int _hitsLanded = 0;
        private int _hitsReceived = 0;

        private SumoAPI api;
        private UltimateState state = UltimateState.Hunt;

        #region Lifecycle

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            state = UltimateState.Hunt;
            _hitsLanded = 0;
            _hitsReceived = 0;
        }

        public override void OnBotUpdate()
        {
            ClearCommands();

            float myEdge = EdgeRatio();
            float enemyEdge = EnemyEdgeRatio();

            if (myEdge >= EDGE_ENTER)
            {
                state = UltimateState.Recover;
            }
            else if (enemyEdge >= ENEMY_EDGE_RANBU && myEdge < EDGE_CAUTION)
            {
                state = UltimateState.Ranbu;
            }
            else if (state == UltimateState.Ranbu && enemyEdge < ENEMY_EDGE_EXIT)
            {
                state = UltimateState.Hunt;
            }

            switch (state)
            {
                case UltimateState.Hunt:    HandleHunt();    break;
                case UltimateState.Attack:  HandleAttack();  break;
                case UltimateState.Ranbu:   HandleRanbu();   break;
                case UltimateState.Recover: HandleRecover(); break;
            }

            Submit();
        }

        public override void OnBotCollision(BounceEvent e)
        {
            if (e.Actor == api.MyRobot.Side)
            {
                _hitsLanded++;
                return;
            }

            _hitsReceived++;

            if (EdgeRatio() >= EDGE_ENTER)
                state = UltimateState.Recover;
        }

        public override void OnBattleStateChanged(BattleState s, BattleWinner? w)
        {
            ClearCommands();
            state = UltimateState.Hunt;
            _hitsLanded = 0;
            _hitsReceived = 0;
        }

        #endregion

        #region State Handlers

        private void HandleHunt()
        {
            Vector2 predicted = PredictedEnemyPosition(PREDICT_TIME);
            float anglePred = AngleToPosition(predicted);
            float angleNow = api.Angle();
            float dist = EnemyDist();

            if (dist < ATTACK_DIST && Mathf.Abs(angleNow) < AggressiveAngle(ATTACK_ANGLE))
            {
                state = UltimateState.Attack;
                HandleAttack();
                return;
            }

            if (!IsPositionDominant())
                TurnToward(ComputeDominanceArcAngle());
            else
                TurnToward(anglePred);

            Enqueue(new AccelerateAction(InputType.Script));

            if (!api.MyRobot.Skill.IsSkillOnCooldown
                && Mathf.Abs(angleNow) < AggressiveAngle(ATTACK_ANGLE)
                && IsSafeToBoost())
                Enqueue(new SkillAction(InputType.Script, ActionType.SkillBoost));
        }

        private void HandleAttack()
        {
            float angleNow = api.Angle();
            float dist = EnemyDist();

            if (dist > EXIT_DIST || Mathf.Abs(angleNow) > EXIT_ANGLE)
            {
                state = UltimateState.Hunt;
                HandleHunt();
                return;
            }

            Vector2 predicted = PredictedEnemyPosition(PREDICT_TIME);
            float anglePred = AngleToPosition(predicted);
            float dynDash = AggressiveAngle(DASH_ANGLE);
            float dynSkill = AggressiveAngle(SKILL_ANGLE);

            TurnToward(anglePred);

            if (!api.MyRobot.IsDashOnCooldown
                && Mathf.Abs(anglePred) < dynDash
                && IsDashSafe()
                && HasApproachMomentum())
                Enqueue(new DashAction(InputType.Script));

            if (!api.MyRobot.Skill.IsSkillOnCooldown
                && Mathf.Abs(anglePred) < dynSkill
                && IsSafeToBoost())
                Enqueue(new SkillAction(InputType.Script, ActionType.SkillBoost));

            Enqueue(new AccelerateAction(InputType.Script));
        }

        private void HandleRanbu()
        {
            Vector2 predicted = PredictedEnemyPosition(PREDICT_TIME);
            float anglePred = AngleToPosition(predicted);

            TurnToward(anglePred);

            if (!api.MyRobot.IsDashOnCooldown
                && Mathf.Abs(anglePred) < RANBU_DASH_ANGLE
                && IsDashSafe())
                Enqueue(new DashAction(InputType.Script));

            if (!api.MyRobot.Skill.IsSkillOnCooldown
                && Mathf.Abs(anglePred) < RANBU_SKILL_ANGLE
                && IsSafeToBoost())
                Enqueue(new SkillAction(InputType.Script, ActionType.SkillBoost));

            Enqueue(new AccelerateAction(InputType.Script));
        }

        private void HandleRecover()
        {
            float edge = EdgeRatio();

            if (edge < EDGE_EXIT)
            {
                state = UltimateState.Hunt;
                HandleHunt();
                return;
            }

            if (edge >= EDGE_DESPERATION)
            {
                if (!api.MyRobot.Skill.IsSkillOnCooldown)
                    Enqueue(new SkillAction(InputType.Script, ActionType.SkillStone));

                TurnToward(api.Angle());

                if (!api.MyRobot.IsDashOnCooldown)
                    Enqueue(new DashAction(InputType.Script));

                Enqueue(new AccelerateAction(InputType.Script));
                return;
            }

            if (!api.MyRobot.Skill.IsSkillOnCooldown)
                Enqueue(new SkillAction(InputType.Script, ActionType.SkillStone));

            TurnToward(AngleToCenter());

            if (IsFacingCenter(0.15f))
                Enqueue(new AccelerateAction(InputType.Script));
        }

        #endregion

        #region Systems

        private Vector2 PredictedEnemyPosition(float time) =>
            api.EnemyRobot.Position + api.EnemyRobot.LinearVelocity * time;

        private float AngleToPosition(Vector2 targetPos)
        {
            Vector2 toTarget = (targetPos - api.MyRobot.Position).normalized;
            Vector2 forward = Quaternion.Euler(0, 0, api.MyRobot.Rotation) * Vector2.up;
            return Vector2.SignedAngle(forward, toTarget);
        }

        private bool IsPositionDominant()
        {
            Vector2 center = api.BattleInfo.ArenaPosition;
            Vector2 myPos = api.MyRobot.Position;
            Vector2 enemyPos = api.EnemyRobot.Position;
            Vector2 centerToMe = (myPos - center).normalized;
            Vector2 meToEnemy = (enemyPos - myPos).normalized;
            return Vector2.Dot(centerToMe, meToEnemy) > POSITION_DOT_THRESHOLD;
        }

        private float ComputeDominanceArcAngle()
        {
            Vector2 center = api.BattleInfo.ArenaPosition;
            Vector2 enemyPos = api.EnemyRobot.Position;
            Vector2 enemyToCenter = (center - enemyPos).normalized;
            Vector2 idealPos = enemyPos + enemyToCenter * (ATTACK_DIST * 0.8f);
            return AngleToPosition(idealPos);
        }

        private bool HasApproachMomentum()
        {
            Vector2 toEnemy = (api.EnemyRobot.Position - api.MyRobot.Position).normalized;
            Vector2 velocity = api.MyRobot.LinearVelocity;
            return Vector2.Dot(velocity.normalized, toEnemy) > 0.5f
                   && velocity.magnitude > api.MyRobot.MoveSpeed * DASH_MOMENTUM_RATIO;
        }

        private float AggressiveAngle(float baseAngle)
        {
            if (_hitsReceived == 0) return baseAngle;
            float ratio = (float)_hitsLanded / _hitsReceived;
            if (ratio >= 1f) return baseAngle;
            float deficit = Mathf.Clamp01(1f - ratio);
            return baseAngle * (1f + deficit * 0.4f);
        }

        private bool IsSafeToBoost() => EdgeRatio() < EDGE_CAUTION;

        private float EnemyEdgeRatio() =>
            api.Distance(
                targetPos: api.EnemyRobot.Position,
                oriPos: api.BattleInfo.ArenaPosition
            ).magnitude / api.BattleInfo.ArenaRadius;

        #endregion

        #region Helpers

        private void TurnToward(float angle)
        {
            if (Mathf.Abs(angle) <= FOCUS_ANGLE) return;
            float dur = Mathf.Max(0.05f, Mathf.Abs(angle) / api.MyRobot.RotateSpeed);
            dur = Mathf.Max(dur, 0.1f); // Ensure a minimum duration for better responsiveness
			Enqueue(new TurnAction(InputType.Script,
                angle > 0 ? ActionType.TurnLeft : ActionType.TurnRight, dur));
        }

        private float AngleToCenter()
        {
            Vector2 toCenter = (api.BattleInfo.ArenaPosition - api.MyRobot.Position).normalized;
            Vector2 forward = Quaternion.Euler(0, 0, api.MyRobot.Rotation) * Vector2.up;
            return Vector2.SignedAngle(forward, toCenter);
        }

        private float EnemyDist() =>
            Vector2.Distance(api.EnemyRobot.Position, api.MyRobot.Position);

        private float EdgeRatio() =>
            api.Distance(targetPos: api.BattleInfo.ArenaPosition).magnitude
            / api.BattleInfo.ArenaRadius;

        private bool IsFacingCenter(float threshold)
        {
            Vector2 toCenter = (api.BattleInfo.ArenaPosition - api.MyRobot.Position).normalized;
            Vector2 forward = Quaternion.Euler(0, 0, api.MyRobot.Rotation) * Vector2.up;
            return Vector2.Dot(forward, toCenter) > threshold;
        }

        private bool IsDashSafe() => IsFacingCenter(0f);

        #endregion
    }
}
