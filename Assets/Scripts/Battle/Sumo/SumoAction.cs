
using System;
using Unity.VisualScripting;

namespace CoreSumo
{

    public abstract class ISumoAction
    {
        // An InputType that player use for this action.
        // InputUsed is utilized mostly in preventing multiple inputs for one action
        // for example: 
        // 1. A battle started with UI Input Type, keyboard also can be used
        // 2. A battle started with UI Input Type, and Sumo is associated with Script Component [Bot]
        public InputType InputUsed;

        // Why this action is taken. 
        // Typically used by Script, because we have a reason to give our robot an action, good for labelling in the making of AI
        public string Reason;
        
        public object Param;
        public abstract void Execute(SumoController controller);

        public string Name
        {
            get
            {
                // Get "Accelerate" instead of "AccelerateAction"
                var name = GetType().Name;
                return name.EndsWith("Action") ? name.Remove(name.Length - "Action".Length) : name;
            }
        }

        public string NameWithParam
        {
            get
            {
                if (Param == null)
                {
                    return Name;
                }
                return $"{GetType().Name}_{Param}";
            }
        }

        public override string ToString()
        {
            return String.Format($"{InputUsed}:{Param}");
        }
    }

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
        public AccelerateAction(InputType inputType)
        {
            InputUsed = inputType;
        }
        public override void Execute(SumoController controller)
        {
            controller.Accelerate(this, AccelerateActionType.Default);
        }
    }

    public class TurnLeftAction : ISumoAction
    {

        public TurnLeftAction(InputType inputType)
        {
            InputUsed = inputType;
        }

        public override void Execute(SumoController controller)
        {

            controller.Turn(this, TurnActionType.Left);
        }
    }

    public class TurnRightAction : ISumoAction
    {
        public TurnRightAction(InputType inputType)
        {
            InputUsed = inputType;
        }

        public override void Execute(SumoController controller)
        {

            controller.Turn(this, TurnActionType.Right);
        }
    }


    public class DashAction : ISumoAction
    {
        public DashAction(InputType inputType)
        {
            InputUsed = inputType;
        }
        public override void Execute(SumoController controller)
        {
            controller.Dash(this, DashActionType.Default);
        }
    }

    public class SkillAction : ISumoAction
    {
        public SkillAction(InputType inputType)
        {
            InputUsed = inputType;
        }

        public override void Execute(SumoController controller)
        {
            controller.Skill.Activate(this);
        }
    }

    #region Live Command / Script
    public class AccelerateTimeAction : ISumoAction
    {
        public AccelerateTimeAction(float time, InputType inputUsed = InputType.Script)
        {
            Param = time;
            InputUsed = inputUsed;
        }
        public override void Execute(SumoController controller)
        {

            controller.Accelerate(this, AccelerateActionType.Time);
        }
    }

    public class DashTimeAction : ISumoAction
    {

        public DashTimeAction(float time, InputType inputUsed = InputType.Script)
        {
            Param = time;
            InputUsed = inputUsed;
        }
        public override void Execute(SumoController controller)
        {

            controller.Dash(this, DashActionType.Time);
        }
    }

    public class TurnLeftAngleAction : ISumoAction
    {
        public TurnLeftAngleAction(float angle, InputType inputUsed = InputType.Script)
        {
            Param = angle;
            InputUsed = inputUsed;
        }

        public override void Execute(SumoController controller)
        {

            controller.Turn(this);
        }
    }


    public class TurnRightAngleAction : ISumoAction
    {

        public TurnRightAngleAction(float angle, InputType inputUsed = InputType.Script)
        {
            Param = angle;
            InputUsed = inputUsed;
        }

        public override void Execute(SumoController controller)
        {

            controller.Turn(this);
        }
    }


    public class TurnAngleAction : ISumoAction
    {

        public TurnAngleAction(float angle, InputType inputUsed = InputType.Script)
        {
            Param = angle;
            InputUsed = inputUsed;
        }

        public override void Execute(SumoController controller)
        {

            controller.Turn(this);
        }
    }

    #endregion

}