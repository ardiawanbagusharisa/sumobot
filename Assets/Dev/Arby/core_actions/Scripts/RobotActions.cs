using UnityEngine;

namespace RobotCoreAction
{
    public class AccelerateAction : ISumoAction
    {
        // public float Speed { get; }
        // public AccelerateAction(float speed)
        // {
        //     Speed = speed;
        // }

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
        public void Execute(RobotActionController controller, RobotStats stats)
        {
            controller.UseSkill();
        }
    }

    public enum ERobotActionType
    {
        Accelerate,
        Dash,
        Skill,
        Idle,
    }

}