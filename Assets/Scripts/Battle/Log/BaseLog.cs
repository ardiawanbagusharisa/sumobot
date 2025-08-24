
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace SumoLog
{
    public class RobotLog
    {
        public BaseLog Robot = new();
        public BaseLog EnemyRobot = new();
        public Dictionary<string, dynamic> ToMap()
        {
            return new(){
                { "Robot", Robot.ToMap()},
                { "EnemyRobot", EnemyRobot.ToMap() },
            };
        }

        public static RobotLog FromMap(Dictionary<string, object> data)
        {
            var robot = BaseLog.FromObject((JObject)data["Robot"]);
            var enemyRobot = BaseLog.FromObject((JObject)data["EnemyRobot"]);
            return new RobotLog()
            {
                Robot = robot,
                EnemyRobot = enemyRobot,
            };
        }
    }
    public class BaseLog
    {
        public Vector2 Position;
        public float AngularVelocity;
        public float LinearVelocity;
        public float Rotation;
        public bool IsDashActive;
        public bool IsSkillActive;
        public bool IsOutFromArena;

        public Dictionary<string, dynamic> ToMap()
        {
            return new(){
            { "AngularVelocity", AngularVelocity},
            { "LinearVelocity", LinearVelocity },
            { "Position", new Dictionary<string,float>()
                {
                    {"X",Position.x},
                    {"Y",Position.y},
                }
            },
            { "Rotation", Rotation},
            { "IsDashActive", IsDashActive},
            { "IsSkillActive", IsSkillActive},
            { "IsOutFromArena", IsOutFromArena},
        };
        }


        public static BaseLog FromMap(Dictionary<string, object> data)
        {
            try
            {
                var robot = (JObject)data["Robot"];
                return FromObject(robot);
            }
            catch
            {
                var robot = (Dictionary<string, object>)data["Robot"];
                var position = (Dictionary<string, float>)robot["Position"];

                Vector2 temPosition = new(position["X"], position["Y"]);

                BaseLog result = new()
                {
                    AngularVelocity = (float)robot?["AngularVelocity"],
                    LinearVelocity = (float)robot?["LinearVelocity"],
                    Position = temPosition,
                    Rotation = (float)robot?["Rotation"],
                    IsDashActive = (bool)robot?["IsDashActive"],
                    IsSkillActive = (bool)robot?["IsSkillActive"],
                    IsOutFromArena = (bool)robot?["IsOutFromArena"],
                };
                return result;
            }
        }

        public static BaseLog FromObject(JObject robot)
        {
            Vector2 temPosition = new((float)(double)robot["Position"]["X"], (float)(double)robot["Position"]["Y"]);

            BaseLog result = new()
            {
                AngularVelocity = (float)robot?["AngularVelocity"],
                LinearVelocity = (float)(double)robot?["LinearVelocity"],
                Position = temPosition,
                Rotation = (float)(double)robot?["Rotation"],
                IsDashActive = (bool)robot?["IsDashActive"],
                IsSkillActive = (bool)robot?["IsSkillActive"],
                IsOutFromArena = (bool)robot?["IsOutFromArena"],
            };
            return result;
        }
    }
}