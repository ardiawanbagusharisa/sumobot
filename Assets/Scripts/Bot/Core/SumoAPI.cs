using System.Collections.Generic;
using SumoCore;
using SumoManager;
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

        public SumoBotAPI MyRobot
        {
            get
            {
                return new SumoBotAPI(myController);
            }
        }

        public SumoBotAPI EnemyRobot
        {
            get
            {
                return new SumoBotAPI(enemyController);
            }
        }

        public bool IsActionActive(ISumoAction action, bool isEnemy = false)
        {
            if (isEnemy)
                return enemyController.IsActionActive(action.Type);
            else
                return myController.IsActionActive(action.Type);
        }

        public bool CanExecute(ISumoAction action)
        {
            return myController.InputProvider.CanExecute(action);
        }
        
        public void StopCoroutine(Coroutine func)
        {
            myController.StopCoroutine(func);
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
            return dist.magnitude / (2 * BattleInfo.ArenaRadius);
        }

        // Return amount of degree from [original] to [target]
        // 0 or 360 -> Up
        // 270  -> right
        // 180 -> bottom
        // 90 -> left
        public float AngleDeg(
            Vector2? oriPos = null,
            float? oriRot = null,
            Vector2? targetPos = null)
        {
            Vector2 toEnemy = Distance(oriPos, targetPos);

            float angleToEnemy = Mathf.Atan2(toEnemy.y, toEnemy.x) * Mathf.Rad2Deg - 90f;
            if (angleToEnemy < 0) angleToEnemy += 360f;

            float relativeAngle = angleToEnemy - (oriRot ?? MyRobot.Rotation);
            if (relativeAngle < 0) relativeAngle += 360f;

            relativeAngle = (relativeAngle + 360f) % 360f;

            return relativeAngle;
        }

        public float Angle(
            Vector2? oriPos = null,
            float? oriRot = null,
            Vector2? targetPos = null,
            bool normalized = false)
        {
            var zRot = oriRot ?? MyRobot.Rotation % 360f;
            if (zRot < 0) zRot += 360f;

            Vector2 facingDir = Quaternion.Euler(0, 0, zRot) * Vector2.up;
            Vector2 toTarget = Distance(oriPos, targetPos).normalized;

            float signedAngle = Vector2.SignedAngle(facingDir, toTarget);

            if (normalized)
                return Mathf.Cos(signedAngle * Mathf.Deg2Rad);
            else
                return signedAngle;
        }

        public (Vector2, float) Simulate(
            List<ISumoAction> actions,
            bool isEnemy = false)
        {
            SumoBotAPI robot = isEnemy ? EnemyRobot : MyRobot;
            float moveSpeed = robot.MoveSpeed;
            float dashSpeed = robot.DashSpeed;
            Vector2 position = robot.Position;
            float rotation = robot.Rotation % 360;
            if (rotation < 0) rotation += 360f;

            foreach (ISumoAction action in actions)
            {
                if (action is SkillAction)
                {
                    if (action.Type == ActionType.SkillBoost)
                    {
                        moveSpeed *= robot.Skill.BoostMultiplier;
                        dashSpeed *= robot.Skill.BoostMultiplier;
                    }
                    if (action.Type == ActionType.SkillStone)
                    {
                        moveSpeed = 0;
                        dashSpeed = 0;
                    }
                }

                if (action is TurnAction)
                {
                    float delta = robot.RotateSpeed * action.Duration;

                    if (action.Type == ActionType.TurnRight)
                        delta = -delta;

                    rotation += delta;
                }

                Vector2 direction = Quaternion.Euler(0, 0, rotation) * Vector2.up;

                if (moveSpeed == 0 && dashSpeed == 0)
                    continue;

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
    public int BestOf { get; }
    public int CurrentRound { get; }
    public int LeftWinCount { get; }
    public int RightWinCount { get; }
    public BattleState CurrentState { get; }
    public float ArenaRadius { get; }
    public Vector2 ArenaPosition { get; }

    public BattleInfoAPI(BattleManager manager)
    {
        TimeLeft = manager.TimeLeft;
        Duration = manager.BattleTime;
        CurrentState = manager.CurrentState;
        BestOf = (int)manager.RoundSystem;
        CurrentRound = manager.Battle.CurrentRound.RoundNumber;
        LeftWinCount = manager.Battle.LeftWinCount;
        RightWinCount = manager.Battle.RightWinCount;

        ArenaPosition = manager.Arena.transform.position;
        ArenaRadius = manager.ArenaRadius;
    }

    public override string ToString()
    {
        return $"[Battle]\n" +
               $"- Time Left        : {TimeLeft:F2}s\n" +
               $"- Duration         : {Duration:F2}s\n" +
               $"- CurrentState     : {CurrentState}\n" +
               $"- BestOf           : {BestOf}\n" +
               $"- CurrendRound     : {CurrentRound}\n" +
               $"- LeftWinCount     : {LeftWinCount}\n" +
               $"- RightWinCount    : {RightWinCount}\n" +
               $"- RightWinCount    : {RightWinCount}";
    }
}

public readonly struct SumoBotAPI
{
    public PlayerSide Side { get; }
    public float MoveSpeed { get; }
    public float RotateSpeed { get; }
    public float DashSpeed { get; }
    public float DashDuration { get; }
    public float DashCooldown { get; }

    public float StopDelay { get; }
    public float BounceResistance { get; }

    public Vector2 Position { get; }
    public float Rotation { get; }
    public Vector2 LinearVelocity { get; }
    public float AngularVelocity { get; }
    public SumoSkillAPI Skill { get; }

    public bool IsDashOnCooldown { get; }
    public bool IsDashActive { get; }
    public bool IsMovementDisabled { get; }
    public bool IsOutFromArena { get; }
    public Dictionary<ActionType, float> ActiveActions { get; }

    public SumoBotAPI(SumoController controller)
    {
        Side = controller.Side;
        MoveSpeed = controller.MoveSpeed;
        RotateSpeed = controller.RotateSpeed;
        DashSpeed = controller.DashSpeed;
        DashDuration = controller.DashDuration;
        DashCooldown = controller.DashCooldown;
        StopDelay = controller.StopDelay;
        BounceResistance = controller.BounceResistance;
        Skill = new(controller.Skill);

        Position = controller.RigidBody.position;
        Rotation = controller.RigidBody.rotation;
        LinearVelocity = controller.RigidBody.linearVelocity;
        AngularVelocity = controller.RigidBody.angularVelocity;

        IsDashOnCooldown = controller.IsDashOnCooldown;
        IsDashActive = controller.IsDashActive;
        IsMovementDisabled = controller.IsMovementDisabled;
        IsOutFromArena = controller.IsOutOfArena;

        ActiveActions = controller.ActiveActions;
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
               $"- StopDelay     : {StopDelay:F2}, BounceResist: {BounceResistance:F2}\n" +
               $"- IsDashOnCooldown   : {IsDashOnCooldown}\n" +
               $"- IsMovementLock: {IsMovementDisabled}\n" +
               $"- Skill         : {Skill.ToString() ?? "None"}";
    }
}

public readonly struct SumoSkillAPI
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

    public SumoSkillAPI(SumoSkill skill)
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