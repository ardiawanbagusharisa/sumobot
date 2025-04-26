
using System;

namespace CoreSumoRobot
{

    public enum TurnActionType
    {
        Left,
        Right,
        Angle,
    }

    public enum AccelerateActionType
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
            controller.Dash();
        }
    }

    public class TurnLeftAction : ISumoAction
    {

        public void Execute(SumoRobotController controller)
        {
            controller.Turn(TurnActionType.Left);
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
        public ISkill Skill { get; }

        public SkillAction(ISkill skill)
        {
            Skill = skill;
        }


        public void Execute(SumoRobotController controller)
        {
            controller.UseSkill(Skill);
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