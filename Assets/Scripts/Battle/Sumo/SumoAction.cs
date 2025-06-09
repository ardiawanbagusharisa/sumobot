
using System;

namespace CoreSumo
{

    public abstract class ISumoAction
    {
        // InputUsed is utilized mostly in preventing multiple inputs for one action
        // for example: 
        // 1. A battle started with UI Input Type, keyboard also can be used
        // 2. A battle started with UI Input Type, and Sumo is associated with Script Component [Bot]
        public InputType InputUsed;
        public object Param;
        public abstract void Execute(SumoController controller);

        public string Name
        {
            get
            {
                if (Param == null)
                {
                    return GetType().Name;
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
            controller.Accelerate(AccelerateActionType.Default);
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
            controller.Turn(TurnActionType.Left);
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
            controller.Turn(TurnActionType.Right);
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
            controller.Dash(DashActionType.Default);
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
            controller.UseSkill();
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
            controller.Accelerate(AccelerateActionType.Time, (float)Param);
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
            controller.Dash(DashActionType.Time, (float)Param);
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
            controller.Turn(TurnActionType.LeftAngle, (float)Param);
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
            controller.Turn(TurnActionType.RightAngle, (float)Param);
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
            controller.Turn(TurnActionType.Angle, (float)Param);
        }
    }

    #endregion

}