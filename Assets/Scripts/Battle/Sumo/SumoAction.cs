using System;
using SumoInput;
using UnityEngine;
using Newtonsoft.Json;

namespace SumoCore
{
    #region Action abstract class and Enums 

    [Serializable]
    public abstract class ISumoAction
    {
        [JsonIgnore]
        public const float MinDuration = 0.1f;

        [JsonIgnore]
        public InputType InputUsed;

        [JsonIgnore]
        public string Reason;

        // Duration will be replaced for Dash and Skill
        public float Duration = MinDuration;
        public ActionType Type;

        public abstract void Execute(SumoController controller);

        [JsonIgnore]
        public string Name
        {
            get
            {
                return $"{Type}";
            }
        }

        [JsonIgnore]
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
            controller.Accelerate(this);
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