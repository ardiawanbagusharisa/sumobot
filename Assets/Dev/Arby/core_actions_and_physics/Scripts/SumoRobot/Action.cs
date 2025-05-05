
using System;

namespace CoreSumoRobot
{

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
        public void Execute(SumoRobotController controller)
        {
            controller.Accelerate(AccelerateActionType.Default);
        }
    }


    public class AccelerateTimeAction : ISumoAction
    {
        public float Time { get; }

        public AccelerateTimeAction(float time)
        {
            Time = time;
        }
        public void Execute(SumoRobotController controller)
        {
            controller.Accelerate(AccelerateActionType.Time, Time);
        }
    }

    public class DashAction : ISumoAction
    {
        public void Execute(SumoRobotController controller)
        {
            controller.Dash(DashActionType.Default);
        }
    }


    public class DashTimeAction : ISumoAction
    {
        public float Time { get; }

        public DashTimeAction(float time)
        {
            Time = time;
        }
        public void Execute(SumoRobotController controller)
        {
            controller.Dash(DashActionType.Time, Time);
        }
    }

    public class TurnLeftAction : ISumoAction
    {

        public void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.Left);
        }
    }

    public class TurnLeftAngleAction : ISumoAction
    {
        public float AngleValue { get; }

        public TurnLeftAngleAction(float angle)
        {
            AngleValue = angle;
        }

        public void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.LeftAngle);
        }
    }

    public class TurnRightAction : ISumoAction
    {
        public string Description => "";

        public void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.Right);
        }
    }

    public class TurnRightAngleAction : ISumoAction
    {
        public float AngleValue { get; }

        public TurnRightAngleAction(float angle)
        {
            AngleValue = angle;
        }

        public void Execute(SumoRobotController controller)
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

        public void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.Angle, AngleValue);
        }
    }

    public class SkillAction : ISumoAction
    {
        public ERobotSkillType ERobotSkillType { get; }

        public SkillAction(ERobotSkillType skillType)
        {
            ERobotSkillType = skillType;
        }


        public void Execute(SumoRobotController controller)
        {
            controller.UseSkill(ERobotSkillType);
        }
    }

    public enum ERobotActionType
    {
        Accelerate,
        Dash,
        Skill,
        Idle,
    }

    public interface ISumoAction
    {
        void Execute(SumoRobotController controller);
    }
}