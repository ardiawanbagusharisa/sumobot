using System;

namespace CoreSumo
{
    public abstract class ISumoAction
    {
        public InputType InputUsed;
        public object Param;
        public abstract void Execute(SumoController controller);

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

    public class AccelerateAction : ISumoAction
    {
        public AccelerateAction(InputType inputType)
        {
            InputUsed = inputType;
        }
        public override void Execute(SumoController controller)
        {
            controller.Accelerate(AccelerateType.Default);
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
            controller.Turn(TurnType.Left);
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
            controller.Turn(TurnType.Right);
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
            controller.Dash(DashType.Default);
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
            controller.Accelerate(AccelerateType.Time, (float)Param);
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
            controller.Dash(DashType.Time, (float)Param);
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
            controller.Turn(TurnType.LeftAngle, (float)Param);
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
            controller.Turn(TurnType.RightAngle, (float)Param);
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
            controller.Turn(TurnType.Angle, (float)Param);
        }
    }

    #endregion

}