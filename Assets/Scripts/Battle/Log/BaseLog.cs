
using System.Collections.Generic;
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
            { "Rotation", new Dictionary<string,float>()
                {
                    {"Z",Rotation},
                }
            },
        };
        }
    }
}