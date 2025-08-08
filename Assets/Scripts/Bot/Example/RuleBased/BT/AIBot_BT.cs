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
        public override string ID => "BT_Bot";
        public override SkillType SkillType => SkillType.Boost;
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
        #endregion

        #region Runtime Properties
        private SumoAPI api;
        public BTNode root;
        #endregion

        #region Bot Template Methods
        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;

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
            root.Tick();
            Submit();
        }

        public override void OnBotCollision(BounceEvent param)
        {
            // Could integrate a temporary BT override for collision reaction
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
            return angle < approachAngle && dist < approachDistance;
        }

        private bool IsEnemyInAttackRange()
        {
            float angle = Mathf.Abs(api.Angle());
            float dist = Vector3.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
            return angle < attackAngle && dist < attackDistance;
        }
        #endregion

        #region Actions
        private BTNode.State Search()
        {
            float angle = api.Angle();
            float turnAmount = Mathf.Clamp(Mathf.Abs(angle), 5f, 30f);
            float duration = Mathf.Max(turnAmount / api.MyRobot.RotateSpeed * api.MyRobot.TurnRate, minTurnDuration);

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
                float dur = Mathf.Max(Mathf.Abs(angle) / api.MyRobot.RotateSpeed * api.MyRobot.TurnRate, minTurnDuration);
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
