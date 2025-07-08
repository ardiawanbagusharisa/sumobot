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
        public float Duration;

        public new Dictionary<string, dynamic> ToMap()
        {
            return new()
            {
                { "Name", Action.Name},
                { "Parameter", Action.Param},
                { "Reason", Action.Reason},

                { "Duration", Duration},
                { "Robot", base.ToMap()},
            };
        }
        public static new ActionLog FromMap(Dictionary<string, object> map)
        {
            var robot = (JObject)map["Robot"];
            Vector2 tempLinearVelocity = new((float)(double)robot["LinearVelocity"]["X"], (float)(double)robot["LinearVelocity"]["Y"]);
            Vector2 temPosition = new((float)(double)robot["Position"]["X"], (float)(double)robot["LinearVelocity"]["Y"]);

            var name = (string)map?["Name"];
            var param = (float?)(double?)map?["Parameter"] ?? null;

            Debug.Log($"name {name} | {param}");
            ActionLog result = new()
            {
                Action = ActionFactory.Parse(name, param),
                Duration = (float)(double)map?["Duration"],
                AngularVelocity = (float)robot?["AngularVelocity"],
                LinearVelocity = tempLinearVelocity,
                Position = temPosition,
                Rotation = (float)(double)robot?["Rotation"]?["Z"]
            };
            return result;
        }

        public static class ActionFactory
        {
            public static ISumoAction Parse(string name, float? parameter)
            {
                if (string.IsNullOrEmpty(name))
                    return null;

                var parts = name.Split('_');
                if (parts.Length != 2)
                    return null;

                string action = parts[1];    // e.g., "TurnLeftWithAngle"

                if (Enum.TryParse(action, out ActionType type))
                {
                    switch (type)
                    {
                        case ActionType.TurnLeftWithAngle:
                            return new TurnAction(InputType.Script, ActionType.TurnLeftWithAngle, parameter ?? 0f);
                        case ActionType.TurnRightWithAngle:
                            return new TurnAction(InputType.Script, ActionType.TurnRightWithAngle, parameter ?? 0f);
                        case ActionType.Accelerate:
                            return new AccelerateAction(InputType.Script);
                        case ActionType.AccelerateWithTime:
                            return new AccelerateAction(InputType.Script, parameter ?? 0f);
                        case ActionType.Dash:
                            return new DashAction(InputType.Script);
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