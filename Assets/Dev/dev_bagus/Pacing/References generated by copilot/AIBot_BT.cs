using SumoCore;
using SumoInput;
using SumoManager;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace SumoBot
{
    public class AIBot_BT : Bot
    {
        #region Runtime Properties
        public override string ID => "Bot_BT";
        public override SkillType DefaultSkillType => SkillType.Boost;
        #endregion

        #region Behavior Tree Parameters
        [Header("BT Parameters")]
        public float approachAngle = 20f;
        public float attackAngle = 15f;
        public float dashSkillAngle = 10f;
        public float approachDistance = 5.0f;
        public float attackDistance = 2.5f;
        public float minTurnDuration = 0.1f;
        public float minAccelerationRatio = 0.9f;

        [Header("Pacing Overlay")]
        public PacingProfile pacingProfile;
        [Range(0f, 1f)] public float pacingAggression = 0.35f;
        #endregion

        #region Runtime Properties
        private SumoAPI api;
        public BTNode root;

        private PacingController pacingController;
        private PacingFrame pacingFrame;

        private float baseApproachAngle;
        private float baseAttackAngle;
        private float baseApproachDistance;
        private float baseAttackDistance;

        private float tunedApproachAngle;
        private float tunedAttackAngle;
        private float tunedApproachDistance;
        private float tunedAttackDistance;
        #endregion

        #region Bot Template Methods
        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;

            baseApproachAngle = approachAngle;
            baseAttackAngle = attackAngle;
            baseApproachDistance = approachDistance;
            baseAttackDistance = attackDistance;

            tunedApproachAngle = approachAngle;
            tunedAttackAngle = attackAngle;
            tunedApproachDistance = approachDistance;
            tunedAttackDistance = attackDistance;

            // Auto-load pacing profile if not assigned
            if (pacingProfile == null)
            {
                pacingProfile = LoadPacingProfile();
            }

            if (pacingProfile != null)
            {
                pacingController = new PacingController(pacingProfile);
                pacingController.Init(api);
            }

            root = new Selector(
                new Sequence(
                    new ConditionNode(IsEnemyInAttackRange),
                    new ActionNode(Attack)
                ),
                new Sequence(
                    new ConditionNode(IsEnemyInApproachRange),
                    new ActionNode(Approach)
                ),
                new ActionNode(Search),
                new ActionNode(Idle)
            );
        }

        public override void OnBotUpdate()
        {
            ClearCommands();

            if (pacingController != null)
            {
                pacingFrame = pacingController.Tick();
                UpdatePacingTuning();
            }

            root.Tick();
            Submit();
        }

        public override void OnBotCollision(BounceEvent param)
        {
            pacingController?.RegisterCollision(param);
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
            // Optional resets or emergency logic
        }
        #endregion

        #region Conditions

        private bool IsEnemyVisible()
        {
            float angle = Mathf.Abs(api.Angle());
            float dist = Vector3.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
            return angle < 90f && dist < 10f;
        }

        private bool IsEnemyInApproachRange()
        {
            float angle = Mathf.Abs(api.Angle());
            float dist = Vector3.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
            return angle < tunedApproachAngle && dist < tunedApproachDistance;
        }

        private bool IsEnemyInAttackRange()
        {
            float angle = Mathf.Abs(api.Angle());
            float dist = Vector3.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
            return angle < tunedAttackAngle && dist < tunedAttackDistance;
        }
        #endregion

        #region Actions
        private BTNode.State Search()
        {
            float angle = api.Angle();
            float turnAmount = Mathf.Clamp(Mathf.Abs(angle), 5f, 30f);
            float duration = Mathf.Max(turnAmount / api.MyRobot.RotateSpeed, minTurnDuration);

            ActionType turn = angle >= 0 ? ActionType.TurnLeft : ActionType.TurnRight;
            Enqueue(new TurnAction(InputType.Script, turn, duration));

            if (api.MyRobot.LinearVelocity.magnitude < api.MyRobot.MoveSpeed * minAccelerationRatio)
                Enqueue(new AccelerateAction(InputType.Script));

            return BTNode.State.Success;
        }

        private BTNode.State Approach()
        {
            float angle = api.Angle();
            float distance = Vector3.Distance(api.EnemyRobot.Position, api.MyRobot.Position);

            if (Mathf.Abs(angle) > 3f)
            {
                float dur = Mathf.Max(Mathf.Abs(angle) / api.MyRobot.RotateSpeed, minTurnDuration);
                ActionType turn = angle > 0 ? ActionType.TurnLeft : ActionType.TurnRight;
                Enqueue(new TurnAction(InputType.Script, turn, dur));
            }

            if (api.MyRobot.LinearVelocity.magnitude < api.MyRobot.MoveSpeed * minAccelerationRatio)
                Enqueue(new AccelerateAction(InputType.Script));

            return BTNode.State.Success;
        }

        private BTNode.State Attack()
        {
            float angle = Mathf.Abs(api.Angle());

            if (!api.MyRobot.IsDashOnCooldown && angle < dashSkillAngle)
                Enqueue(new DashAction(InputType.Script));

            if (!api.MyRobot.Skill.IsSkillOnCooldown && angle < dashSkillAngle)
                Enqueue(new SkillAction(InputType.Script));

            if (api.MyRobot.LinearVelocity.magnitude < api.MyRobot.MoveSpeed * minAccelerationRatio)
                Enqueue(new AccelerateAction(InputType.Script));

            return BTNode.State.Success;
        }

        private BTNode.State Idle()
        {
            return BTNode.State.Success; // do nothing for now
        }
        #endregion
        #region Pacing helpers
        private PacingProfile LoadPacingProfile()
        {
            // Extract bot suffix from ID (e.g., "Bot_BT" -> "BT")
            string botSuffix = ID.Replace("Bot_", "");
            string resourcePath = $"Pacing/Pacing_{botSuffix}";
            
            PacingProfile profile = Resources.Load<PacingProfile>(resourcePath);
            if (profile != null)
                Debug.Log($"[{ID}] Loaded pacing profile: {resourcePath}");
            else
                Debug.Log($"[{ID}] Pacing profile not found at: Assets/Resources/{resourcePath}.asset");
            
            return profile;
        }

        private void UpdatePacingTuning()
        {
            float target = pacingFrame.target;
            float current = pacingFrame.overall;

            float aggression = Mathf.Clamp01(0.5f + (target - current) * pacingAggression);

            tunedApproachDistance = Mathf.Lerp(baseApproachDistance * 1.25f, baseApproachDistance * 0.7f, aggression);
            tunedAttackDistance = Mathf.Lerp(baseAttackDistance * 1.25f, baseAttackDistance * 0.6f, aggression);

            tunedApproachAngle = Mathf.Lerp(baseApproachAngle * 1.2f, baseApproachAngle * 0.65f, aggression);
            tunedAttackAngle = Mathf.Lerp(baseAttackAngle * 1.2f, baseAttackAngle * 0.65f, aggression);
        }

        public override void Enqueue(ISumoAction action)
        {
            pacingController?.RegisterAction(action);
            base.Enqueue(action);
        }
        #endregion
    }

    #region BT Core
    public abstract class BTNode
    {
        public enum State { Success, Failure, Running }
        public abstract State Tick();
    }

    public class Selector : BTNode
    {
        public List<BTNode> children;
        public Selector(params BTNode[] nodes) => children = new List<BTNode>(nodes);

        public override State Tick()
        {
            foreach (var child in children)
            {
                var result = child.Tick();
                if (result != State.Failure)
                    return result;
            }
            return State.Failure;
        }
    }

    public class Sequence : BTNode
    {
        public List<BTNode> children;
        public Sequence(params BTNode[] nodes) => children = new List<BTNode>(nodes);

        public override State Tick()
        {
            foreach (var child in children)
            {
                var result = child.Tick();
                if (result != State.Success)
                    return result;
            }
            return State.Success;
        }
    }

    public class ConditionNode : BTNode
    {
        private Func<bool> condition;
        public ConditionNode(Func<bool> condition) => this.condition = condition;
        public override State Tick() => condition() ? State.Success : State.Failure;
    }

    public class ActionNode : BTNode
    {
        private Func<State> action;
        public ActionNode(Func<State> action) => this.action = action;
        public override State Tick() => action();
    }
    #endregion
}
