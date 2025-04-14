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
        List<ISumoAction> GetInput();
    }

    public interface ISkill
    {
        //[Todo] Define skill properties and methods
    }
}