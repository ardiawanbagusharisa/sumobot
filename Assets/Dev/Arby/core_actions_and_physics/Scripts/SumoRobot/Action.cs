
using Unity.VisualScripting;

namespace CoreSumoRobot
{

    public abstract class ISumoAction
    {
        public InputType InputUsed;
        public abstract void Execute(SumoRobotController controller);
    }

    public enum TurnActionType
    {
        Left,
        Right,
        LeftAngle,
        RightAngle,
        Angle,
    }

    public enum AccelerateActionType
    {
        Default,
        Time,
    }

    public enum DashActionType
    {
        Default,
        Time,
    }

    public class AccelerateAction : ISumoAction
    {
        public AccelerateAction(InputType inputType)
        {
            InputUsed = inputType;
        }
        public override void Execute(SumoRobotController controller)
        {
            controller.Accelerate(AccelerateActionType.Default);
        }
    }

    public class TurnLeftAction : ISumoAction
    {

        public TurnLeftAction(InputType inputType)
        {
            InputUsed = inputType;
        }

        public override void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.Left);
        }
    }

    public class TurnRightAction : ISumoAction
    {
        public TurnRightAction(InputType inputType)
        {
            InputUsed = inputType;
        }

        public override void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.Right);
        }
    }


    public class DashAction : ISumoAction
    {
        public DashAction(InputType inputType)
        {
            InputUsed = inputType;
        }
        public override void Execute(SumoRobotController controller)
        {
            controller.Dash(DashActionType.Default);
        }
    }

    public class SkillAction : ISumoAction
    {
        public ERobotSkillType ERobotSkillType { get; }

        public SkillAction(ERobotSkillType skillType, InputType inputType)
        {
            ERobotSkillType = skillType;
            InputUsed = inputType;
        }

        public override void Execute(SumoRobotController controller)
        {
            controller.UseSkill(ERobotSkillType);
        }
    }

    public class AccelerateTimeAction : ISumoAction
    {
        public float Time { get; }

        public AccelerateTimeAction(float time)
        {
            Time = time;
        }
        public override void Execute(SumoRobotController controller)
        {
            controller.Accelerate(AccelerateActionType.Time, Time);
        }
    }

    public class DashTimeAction : ISumoAction
    {
        public float Time { get; }

        public DashTimeAction(float time)
        {
            Time = time;
        }
        public override void Execute(SumoRobotController controller)
        {
            controller.Dash(DashActionType.Time, Time);
        }
    }

    #region Live Command / Script
    public class TurnLeftAngleAction : ISumoAction
    {
        public float AngleValue { get; }

        public TurnLeftAngleAction(float angle)
        {
            AngleValue = angle;
        }

        public override void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.LeftAngle);
        }
    }


    public class TurnRightAngleAction : ISumoAction
    {
        public float AngleValue { get; }

        public TurnRightAngleAction(float angle)
        {
            AngleValue = angle;
        }

        public override void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.RightAngle);
        }
    }


    public class TurnAngleAction : ISumoAction
    {
        public float AngleValue { get; }

        public TurnAngleAction(float angle)
        {
            AngleValue = angle;
        }

        public override void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.Angle, AngleValue);
        }
    }

    #endregion

}