using UnityEngine;
using RobotCoreAction.Controllers;

namespace RobotCoreAction
{

    public class AccelerateAction : ISumoAction
    {

        public void Execute(RobotActionController robot, RobotStats stats)
        {
            robot.Accelerate();
        }
    }

    public class DashAction : ISumoAction
    {
        public void Execute(RobotActionController robot, RobotStats stats)
        {
            stats.ActionsTime.TryGetValue(ERobotActionType.Dash, out float lastActTime);
            if (Time.time >= lastActTime + stats.DashDuration)
            {
                robot.Dash();
            }
            else
            {
                Debug.Log("Dash is on cooldown.");
            }
        }
    }

    public class TurnAction : ISumoAction
    {
        public bool IsRight { get; }

        public TurnAction(bool isRight)
        {
            IsRight = isRight;
        }
        public void Execute(RobotActionController controller, RobotStats stats)
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

        public void Execute(RobotActionController controller, RobotStats stats)
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
        void Execute(RobotActionController controller, RobotStats stats);
    }
}