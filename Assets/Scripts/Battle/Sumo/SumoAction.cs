using SumoInput;
using UnityEngine;

namespace SumoCore
{
    #region Action abstract class and Enums 
    public abstract class ISumoAction
    {
        public const float MinDuration = 0.1f;

        public InputType InputUsed;
        public string Reason;

        [Min(MinDuration)]
        public float Duration = MinDuration;
        public ActionType Type;

        public abstract void Execute(SumoController controller);

        public string Name
        {
            get
            {
                return $"{Type}";
            }
        }

        public string FullName
        {
            get
            {
                if (this is not DashAction || this is not SkillAction)
                    return $"{Name}_{Duration}";
                else
                    return Name;
            }
        }
    }

    public enum ActionType
    {
        Accelerate,
        Dash,
        TurnLeft,
        TurnRight,
        SkillBoost,
        SkillStone,
    }
    #endregion

    #region Action classes
    public class AccelerateAction : ISumoAction
    {

        public AccelerateAction(InputType inputType, float? duration = null)
        {
            InputUsed = inputType;
            Type = ActionType.Accelerate;

            if (duration != null)
                Duration = (float)duration;
        }

        public override void Execute(SumoController controller)
        {
            controller.Accelerate(this);
        }
    }

    public class TurnAction : ISumoAction
    {

        public TurnAction(InputType inputType, ActionType type, float? duration = null)
        {
            Type = type;
            InputUsed = inputType;

            if (duration != null)
                Duration = (float)duration;
        }

        public override void Execute(SumoController controller)
        {
            controller.Turn(this);
        }
    }

    public class DashAction : ISumoAction
    {
        public DashAction(InputType inputType)
        {
            InputUsed = inputType;
            Type = ActionType.Dash;
        }
        public override void Execute(SumoController controller)
        {
            controller.Dash(this);
        }
    }

    public class SkillAction : ISumoAction
    {
        public SkillAction(InputType inputType, ActionType? skillType = null)
        {
            if (skillType != null)
                Type = (ActionType)skillType;
            else
                Type = ActionType.SkillBoost;

            InputUsed = inputType;
        }

        public override void Execute(SumoController controller)
        {
            controller.Skill.Activate(this);
        }
    }
    #endregion
}