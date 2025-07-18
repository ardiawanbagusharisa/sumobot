using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using SumoCore;
using SumoInput;
using Unity.VisualScripting;
using UnityEngine;

namespace SumoLog
{
    public class ActionLog : BaseLog
    {
        [DoNotSerialize]
        public ISumoAction Action;

        public new Dictionary<string, dynamic> ToMap()
        {
            return new()
            {
                { "Name", Action.Name},
                { "Duration", Action.Duration},
                { "Reason", Action.Reason},
                { "Robot", base.ToMap()},
            };
        }
        public static new ActionLog FromMap(Dictionary<string, object> map)
        {
            var robot = (JObject)map["Robot"];
            Vector2 tempLinearVelocity = new((float)(double)robot["LinearVelocity"]["X"], (float)(double)robot["LinearVelocity"]["Y"]);
            Vector2 temPosition = new((float)(double)robot["Position"]["X"], (float)(double)robot["LinearVelocity"]["Y"]);

            var name = (string)map?["Name"];
            var angle = (float?)(double?)map?["Angle"] ?? null;
            var duration = (float?)(double?)map?["Duration"] ?? 0f;

            Debug.Log($"{name} | {duration} | {angle}");
            ActionLog result = new()
            {
                Action = ActionFactory.Parse(name, duration, angle),
                AngularVelocity = (float)robot?["AngularVelocity"],
                LinearVelocity = tempLinearVelocity,
                Position = temPosition,
                Rotation = (float)(double)robot?["Rotation"]?["Z"]
            };
            return result;
        }

        public static class ActionFactory
        {
            public static ISumoAction Parse(string name, float duration, float? angle)
            {
                if (string.IsNullOrEmpty(name))
                    return null;

                var parts = name.Split('_');
                if (parts.Length != 2)
                    return null;

                string action = parts[1];

                if (Enum.TryParse(action, out ActionType type))
                {
                    switch (type)
                    {
                        case ActionType.TurnRight:
                            return new TurnAction(InputType.Script, ActionType.TurnLeft, duration);
                        case ActionType.TurnLeft:
                            return new TurnAction(InputType.Script, ActionType.TurnRight, duration);
                        case ActionType.Accelerate:
                            return new AccelerateAction(InputType.Script, duration);
                        case ActionType.Dash:
                            return new DashAction(InputType.Script);
                        case ActionType.SkillBoost:
                            return new SkillAction(InputType.Script, ActionType.SkillBoost);
                        case ActionType.SkillStone:
                            return new SkillAction(InputType.Script, ActionType.SkillStone);
                        default:
                            Debug.LogWarning($"Unhandled action type: {type}");
                            break;
                    }
                }

                return null;
            }
        }

    }
}