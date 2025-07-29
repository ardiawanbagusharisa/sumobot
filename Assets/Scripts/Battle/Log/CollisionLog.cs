
using System.Collections.Generic;

namespace SumoLog
{
    public class CollisionLog : BaseLog
    {
        public bool IsActor;
        public bool IsTieBreaker = false;
        public float Impact = 0;
        public float BounceResistance;
        public float LockDuration = 0;
        public bool IsSkillActive;
        public bool IsDashActive;

        public float Duration = 0;

        public new Dictionary<string, dynamic> ToMap()
        {
            return new()
            {
                { "IsActor", IsActor},
                { "Impact", Impact},
                { "IsTieBreaker", IsTieBreaker},
                { "LockDuration", LockDuration},
                { "IsSkillActive", IsSkillActive},
                { "IsDashActive", IsDashActive},

                { "Duration", Duration},
                { "Robot", base.ToMap()},
            };
        }
    }
}