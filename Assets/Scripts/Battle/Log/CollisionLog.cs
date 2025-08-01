
using System.Collections.Generic;

namespace SumoLog
{
    public class CollisionLog : RobotLog
    {
        public bool IsActor;
        public bool IsTieBreaker = false;
        public float Impact = 0;
        public float BounceResistance;
        public float LockDuration = 0;

        public float Duration = 0;

        public new Dictionary<string, dynamic> ToMap()
        {
            return new()
            {
                { "IsActor", IsActor},
                { "Impact", Impact},
                { "IsTieBreaker", IsTieBreaker},
                { "LockDuration", LockDuration},

                { "Duration", Duration},
                { "Robot", Robot.ToMap()},
                { "EnemyRobot", EnemyRobot.ToMap()},
            };
        }
    }
}