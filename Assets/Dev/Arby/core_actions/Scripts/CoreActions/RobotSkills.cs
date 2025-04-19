using RobotCoreAction.Controllers;

namespace RobotCoreAction
{
    public interface ISkill
    {
        ERobotSkillType SkillType { get; }
        float Duration { get; }
        float Cooldown { get; }
        void Execute(RobotActionController controller, RobotStats stats, RobotPhysicController physic);
    }
    public enum ERobotSkillType
    {
        Stone,
        Boost
    }
}