using System;

namespace CoreSumo
{
    #region Action abstract class and Enums 
    public abstract class ISumoAction
    {
        public InputType InputUsed;
        public string Reason;
        public object Param;

        public abstract void Execute(SumoController controller);

        public string Name
        {
            get
            {
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

    public enum TurnType
    {
        Left,
        Right,
        LeftAngle,
        RightAngle,
        Angle,
    }

    public enum AccelerateType
    {
        Default,
        Time,
    }

    public enum DashType
    {
        Default,
        Time,
    }
    #endregion

    #region Action classes
    public class AccelerateAction : ISumoAction
    {
        public AccelerateAction(InputType inputType)
        {
            InputUsed = inputType;
        }
        public override void Execute(SumoController controller)
        {
            controller.Accelerate(this, AccelerateType.Default);
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
            controller.Turn(this, TurnType.Left);
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
            controller.Turn(this, TurnType.Right);
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
            controller.Dash(this, DashType.Default);
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
    #endregion

    #region Action classes for Live Command and Script
    public class AccelerateTimeAction : ISumoAction
    {
        public AccelerateTimeAction(float time, InputType inputUsed = InputType.Script)
        {
            Param = time;
            InputUsed = inputUsed;
        }
        public override void Execute(SumoController controller)
        {
            controller.Accelerate(this, AccelerateType.Time);
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
            controller.Dash(this, DashType.Time);
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