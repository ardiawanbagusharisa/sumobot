
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace SumoLog
{
    public class BaseLog
    {
        public float AngularVelocity;
        public Vector2 Position;
        public Vector2 LinearVelocity;
        public float Rotation;

        public Dictionary<string, dynamic> ToMap()
        {
            return new(){
            { "AngularVelocity", AngularVelocity},
            { "LinearVelocity", new Dictionary<string,float>()
                {
                    {"X",LinearVelocity.x},
                    {"Y",LinearVelocity.y},
                }
            },
            { "Position", new Dictionary<string,float>()
                {
                    {"X",Position.x},
                    {"Y",Position.y},
                }
            },
            { "Rotation", Rotation},
        };
        }


        public static BaseLog FromMap(Dictionary<string, object> data)
        {
            var robot = (JObject)data["Robot"];
            Vector2 tempLinearVelocity = new((float)(double)robot["LinearVelocity"]["X"], (float)(double)robot["LinearVelocity"]["Y"]);
            Vector2 temPosition = new((float)(double)robot["Position"]["X"], (float)(double)robot["Position"]["Y"]);

            BaseLog result = new()
            {
                AngularVelocity = (float)robot?["AngularVelocity"],
                LinearVelocity = tempLinearVelocity,
                Position = temPosition,
                Rotation = (float)(double)robot?["Rotation"],
            };
            return result;
        }
    }
}