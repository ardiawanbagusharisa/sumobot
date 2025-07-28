using System.Collections.Generic;
using NUnit.Framework;
using SumoCore;
using SumoManager;
using UnityEditor;
using UnityEngine;

namespace SumoBot
{
    public class SumoAPI
    {
        // [Todo]: Should change the SumoController to exact attribute, giving SumoController directly might be risky
        // Need to configure asmdef (assembly scope)
        private readonly SumoController myController;
        private readonly SumoController enemyController;

        public SumoAPI(SumoController myController, SumoController enemyController)
        {
            this.myController = myController;
            this.enemyController = enemyController;
        }

        public BattleInfoAPI BattleInfo
        {
            get { return new BattleInfoAPI(BattleManager.Instance); }
        }

        public RobotStateAPI MyRobot
        {
            get
            {
                return new RobotStateAPI(myController);
            }
        }

        public RobotStateAPI EnemyRobot
        {
            get
            {
                return new RobotStateAPI(enemyController);
            }
        }

        public bool CanExecute(ISumoAction action)
        {
            return myController.InputProvider.CanExecute(action);
        }

        public Vector2 Distance(
            Vector2? oriPos = null,
            Vector2? targetPos = null)
        {
            return (targetPos ?? EnemyRobot.Position) - (oriPos ?? MyRobot.Position);
        }

        public float DistanceNormalized(
            Vector2? oriPos = null,
            Vector2? targetPos = null)
        {
            Vector2 dist = Distance(oriPos, targetPos);
            return 1f - Mathf.Clamp01(dist.magnitude / BattleInfo.ArenaRadius);
        }

        // Return amount of degree from [original] to [target]
        // 0 or 360 -> Up
        // 270 -> right
        // 180 -> bottom
        // 90 -> left
        public float AngleDeg(
            Vector2? oriPos = null,
            float? oriRot = null,
            Vector2? targetPos = null,
            bool normalized = false)
        {
            Vector2 toEnemy = Distance(oriPos, targetPos);

            float angleToEnemy = Mathf.Atan2(toEnemy.y, toEnemy.x) * Mathf.Rad2Deg - 90f;
            if (angleToEnemy < 0) angleToEnemy += 360f;

            float relativeAngle = angleToEnemy - (oriRot ?? MyRobot.Rotation);
            if (relativeAngle < 0) relativeAngle += 360f;

            relativeAngle = (relativeAngle + 360f) % 360f;

            if (normalized)
                return relativeAngle / 360;
            else
                return relativeAngle;
        }

        public float Angle(
            Vector2? oriPos = null,
            float? oriRot = null,
            Vector2? targetPos = null,
            bool normalized = false)
        {
            Vector2 facingDir = Quaternion.Euler(0, 0, oriRot ?? MyRobot.Rotation) * Vector2.up;
            Vector2 toTarget = Distance(oriPos, targetPos).normalized;

            float signedAngle = Vector2.SignedAngle(facingDir, toTarget);

            if (normalized)
                return Mathf.Cos(signedAngle * Mathf.Deg2Rad);
            else
                return signedAngle;
        }

        public bool IsActionActive(ISumoAction action, bool isEnemy = false)
        {
            var activeActions = isEnemy ? enemyController.ActiveActions : myController.ActiveActions;
            if (activeActions.TryGetValue(action.Type, out float time))
            {
                return time > 0;
            }
            return false;
        }

        public (Vector2, float) Simulate(
            ISumoAction action,
            bool isEnemy = false,
            bool isDelta = false)
        {
            RobotStateAPI robot = isEnemy ? EnemyRobot : MyRobot;
            Vector2 position = robot.Position;
            float rotation = robot.Rotation;

            if (action is TurnAction)
            {
                float delta = robot.RotateSpeed * robot.TurnRate * action.Duration;

                if (action.Type == ActionType.TurnRight)
                    delta = -delta;

                rotation += delta;
            }

            Vector2 direction = Quaternion.Euler(0, 0, rotation) * Vector2.up;

            if (action is AccelerateAction)
            {
                float effectiveSpeed = robot.MoveSpeed;

                float distance = effectiveSpeed * action.Duration;
                position += direction.normalized * distance;
            }
            else if (action is DashAction)
            {
                float effectiveSpeed = robot.DashSpeed;

                float dashDistance = effectiveSpeed * robot.DashDuration;
                position += direction.normalized * dashDistance;
                position += direction.normalized * (robot.StopDelay * effectiveSpeed);
            }

            if (isDelta)
            {
                return new(position - robot.Position, rotation - robot.Rotation);
            }
            return new(position, rotation);
        }

        public override string ToString()
        {
            return $"{BattleInfo}\n\n{EnemyRobot}\n\n{MyRobot}";
        }
    }
}

public readonly struct BattleInfoAPI
{
    public float TimeLeft { get; }
    public float Duration { get; }
    public BattleState CurrentState { get; }
    public float ArenaRadius { get; }
    public Vector2 ArenaPosition { get; }

    public BattleInfoAPI(BattleManager manager)
    {
        TimeLeft = manager.TimeLeft;
        Duration = manager.BattleTime;
        CurrentState = manager.CurrentState;

        ArenaPosition = manager.Arena.transform.position;
        ArenaRadius = manager.ArenaRadius;
    }

    public override string ToString()
    {
        return $"[Battle]\n" +
               $"- Time Left     : {TimeLeft:F2}s\n" +
               $"- Duration      : {Duration:F2}s\n" +
               $"- State         : {CurrentState}";
    }
}

public readonly struct RobotStateAPI
{
    public PlayerSide Side { get; }
    public float MoveSpeed { get; }
    public float RotateSpeed { get; }
    public float DashSpeed { get; }
    public float DashDuration { get; }
    public float DashCooldown { get; }

    public float StopDelay { get; }
    public float TurnRate { get; }
    public float BounceResistance { get; }

    public Vector2 Position { get; }
    public float Rotation { get; }
    public Vector2 LinearVelocity { get; }
    public float AngularVelocity { get; }
    public SkillStateAPI Skill { get; }

    public bool IsDashOnCooldown { get; }
    public bool IsMovementDisabled { get; }

    public RobotStateAPI(SumoController controller)
    {

        Side = controller.Side;
        MoveSpeed = controller.MoveSpeed;
        RotateSpeed = controller.RotateSpeed;
        DashSpeed = controller.DashSpeed;
        DashDuration = controller.DashDuration;
        DashCooldown = controller.DashCooldown;
        TurnRate = controller.TurnRate;
        StopDelay = controller.StopDelay;
        BounceResistance = controller.BounceResistance;
        Skill = new(controller.Skill);

        Position = controller.RigidBody.position;
        Rotation = controller.RigidBody.rotation;
        LinearVelocity = controller.LastLinearVelocity;
        AngularVelocity = controller.RigidBody.angularVelocity;

        IsDashOnCooldown = controller.IsDashOnCooldown;
        IsMovementDisabled = controller.IsMovementDisabled;
    }

    public override string ToString()
    {
        return $"[Robot {Side}]\n" +
               $"- Pos           : {Position}\n" +
               $"- Rot           : {Rotation}\n" +
               $"- Velocity      : {LinearVelocity}\n" +
               $"- AngularVel    : {AngularVelocity:F2}\n" +
               $"- MoveSpeed     : {MoveSpeed:F2}\n" +
               $"- DashSpeed     : {DashSpeed:F2} (Cooldown: {DashCooldown:F2}s)\n" +
               $"- RotateSpeed   : {RotateSpeed:F2}, TurnRate: {TurnRate:F2}\n" +
               $"- StopDelay     : {StopDelay:F2}, BounceResist: {BounceResistance:F2}\n" +
               $"- IsDashOnCooldown   : {IsDashOnCooldown}\n" +
               $"- IsMovementLock: {IsMovementDisabled}\n" +
               $"- Skill         : {Skill.ToString() ?? "None"}";
    }
}

public readonly struct SkillStateAPI
{
    public SkillType Type { get; }
    public float StoneMultiplier { get; }
    public float BoostMultiplier { get; }
    public bool IsActive { get; }
    public bool IsSkillOnCooldown { get; }
    public float TotalCooldown { get; }
    public float TotalDuration { get; }
    public float Cooldown { get; }
    public float CooldownNormalized { get; }

    public SkillStateAPI(SumoSkill skill)
    {
        Type = skill.Type;
        BoostMultiplier = skill.BoostMultiplier;
        StoneMultiplier = skill.StoneMultiplier;
        IsActive = skill.IsActive;
        IsSkillOnCooldown = skill.IsSkillOnCooldown;
        TotalCooldown = skill.TotalCooldown;
        TotalDuration = skill.TotalDuration;
        Cooldown = skill.Cooldown;
        CooldownNormalized = skill.CooldownNormalized;
    }

    public override string ToString()
    {
        string typeLabel = Type.ToString().ToUpper();
        string cooldownStatus = IsSkillOnCooldown ? $"{Cooldown:F1}s ({CooldownNormalized:P0})" : "Ready";
        string activeStatus = IsActive ? "Active" : "Inactive";

        return $"[Skill: {typeLabel}]\n" +
               $"- Status     : {activeStatus}\n" +
               $"- Cooldown   : {cooldownStatus}\n" +
               $"- Duration   : {TotalDuration:F1}s\n" +
               $"- Multiplier : {(Type == SkillType.Boost ? BoostMultiplier : StoneMultiplier):F1}";
    }
}

public readonly struct SimulateResultAPI
{
    public Vector3 Position { get; }
    public Vector3 Rotation { get; }

    public SimulateResultAPI(Vector3 position, Vector3 direction)
    {
        Position = position;
        Rotation = direction;
    }
}