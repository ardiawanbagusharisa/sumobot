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
    public Vector3 ArenaPosition { get; }

    public BattleInfoAPI(BattleManager manager)
    {
        TimeLeft = manager.TimeLeft;
        Duration = manager.BattleTime;
        CurrentState = manager.CurrentState;

        GameObject arena = manager.Arena;
        ArenaPosition = manager.Arena.transform.position;
        ArenaRadius = arena.GetComponent<CircleCollider2D>().radius * arena.transform.lossyScale.x;
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

    public Vector3 Position { get; }
    public Quaternion Rotation { get; }
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

        Position = controller.transform.position;
        Rotation = controller.transform.rotation;
        LinearVelocity = controller.LastLinearVelocity;
        AngularVelocity = controller.LastAngularVelocity;

        IsDashOnCooldown = controller.IsDashOnCooldown;
        IsMovementDisabled = controller.IsMovementDisabled;
    }

    public override string ToString()
    {
        return $"[Robot {Side}]\n" +
               $"- Pos           : {Position}\n" +
               $"- Rot           : {Rotation.eulerAngles}\n" +
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