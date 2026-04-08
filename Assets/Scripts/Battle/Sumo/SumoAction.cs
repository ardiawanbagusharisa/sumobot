using System;
using SumoInput;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace SumoCore
{
    #region Action abstract class and Enums

    [Serializable]
    [JsonConverter(typeof(SumoActionConverter))]
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

        public override string ToString()
        {
            return Name;
        }
    }

    public enum ActionType
    {
        Accelerate, // AccelerateAction
        Dash, // DashAction
        TurnLeft, // TurnAction
        TurnRight, // TurnAction
        SkillBoost, // SkillAction
        SkillStone, // SkillAction
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

    #region JSON Converter
    public class SumoActionConverter : JsonConverter<ISumoAction>
    {
        public override ISumoAction ReadJson(JsonReader reader, Type objectType, ISumoAction existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            JObject jsonObject = JObject.Load(reader);

            // Read the Type field to determine which concrete class to instantiate
            ActionType actionType = jsonObject["Type"].ToObject<ActionType>();
            float duration = jsonObject["Duration"].ToObject<float>();

            ISumoAction action = actionType switch
            {
                ActionType.Accelerate => new AccelerateAction(InputType.Script, duration),
                ActionType.TurnLeft => new TurnAction(InputType.Script, ActionType.TurnLeft, duration),
                ActionType.TurnRight => new TurnAction(InputType.Script, ActionType.TurnRight, duration),
                ActionType.Dash => new DashAction(InputType.Script),
                ActionType.SkillBoost => new SkillAction(InputType.Script, ActionType.SkillBoost),
                ActionType.SkillStone => new SkillAction(InputType.Script, ActionType.SkillStone),
                _ => throw new JsonSerializationException($"Unknown action type: {actionType}")
            };

            // Populate the Duration and other properties from JSON
            serializer.Populate(jsonObject.CreateReader(), action);

            return action;
        }

        public override void WriteJson(JsonWriter writer, ISumoAction value, JsonSerializer serializer)
        {
            // Manually write the JSON to avoid circular reference
            writer.WriteStartObject();
            writer.WritePropertyName("Duration");
            writer.WriteValue(value.Duration);
            writer.WritePropertyName("Type");
            writer.WriteValue(value.Type.ToString());
            writer.WriteEndObject();
        }
    }
    #endregion
}