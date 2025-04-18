using System.Collections.Generic;
using UnityEngine;

namespace RobotCoreAction
{
    public interface ISumoAction
    {
        void Execute(RobotActionController controller, RobotStats stats);
    }

    public interface IInputProvider
    {
        bool IsEnabled { get; }
        List<ISumoAction> GetInput();
    }

    public interface ISkill
    {
        ERobotSkillType SkillType { get; }
        float Duration { get; }
        float Cooldown { get; }
        void Execute(RobotActionController controller, RobotStats stats, RobotPhysicController physic);
    }
}