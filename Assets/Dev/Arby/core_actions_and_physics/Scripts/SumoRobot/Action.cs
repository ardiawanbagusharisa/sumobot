
namespace CoreSumoRobot
{

    public class AccelerateAction : ISumoAction
    {

        public void Execute(SumoRobotController controller)
        {
            controller.Accelerate();
        }
    }

    public class DashAction : ISumoAction
    {
        public void Execute(SumoRobotController controller)
        {
            controller.Dash();
        }
    }

    public class TurnAction : ISumoAction
    {
        public bool IsRight { get; }

        public TurnAction(bool isRight)
        {
            IsRight = isRight;
        }

        public void Execute(SumoRobotController controller)
        {
            controller.Turn(IsRight);
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