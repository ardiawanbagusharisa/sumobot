using SumoInput;

namespace SumoCore
{
    #region Action abstract class and Enums 
    public abstract class ISumoAction
    {
        public InputType InputUsed;
        public string Reason;
        public object Param;
        public ActionType Type;

        public abstract void Execute(SumoController controller);

        public string Name
        {
            get
            {
                var className = GetType().Name;
                var name = className.EndsWith("Action") ? className.Remove(className.Length - "Action".Length) : className;
                return $"{name}_{Type}";
            }
        }

        public string FullName
        {
            get
            {
                if (Param == null)
                {
                    return Name;
                }
                return $"{Name}_({Param})";
            }
        }
    }

    public enum ActionType
    {
        Accelerate,
        AccelerateWithTime,
        Dash,
        TurnLeft,
        TurnRight,
        TurnLeftWithAngle,
        TurnRightWithAngle,
        SkillBoost,
        SkillStone,
    }
    #endregion

    #region Action classes
    public class AccelerateAction : ISumoAction
    {

        public AccelerateAction(InputType inputType, float? time = null)
        {
            InputUsed = inputType;

            if (time != null)
            {
                Type = ActionType.AccelerateWithTime;
                Param = time;
            }
            else
            {
                Type = ActionType.Accelerate;
            }
        }

        public override void Execute(SumoController controller)
        {
            controller.Accelerate(this);
        }
    }

    public class TurnAction : ISumoAction
    {

        public TurnAction(InputType inputType, ActionType type, float? angle = null)
        {
            Param = angle;
            Type = type;
            InputUsed = inputType;
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
            {
                Type = (ActionType)skillType;
            }
            else
            {
                Type = ActionType.SkillBoost;
            }
            InputUsed = inputType;
        }

        public override void Execute(SumoController controller)
        {
            controller.Skill.Activate(this);
        }
    }
    #endregion
}