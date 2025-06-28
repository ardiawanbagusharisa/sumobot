
using System.Collections.Generic;


public class CollisionLog : BaseLog
{
    public bool IsActor;
    public float Impact;
    public float BounceResistance;
    public float LockDuration = float.NaN;
    public bool IsSkillActive;
    public bool IsDashActive;

    public float Duration;

    public new Dictionary<string, dynamic> ToMap()
    {
        return new()
        {
            { "IsActor", IsActor},
            { "Impact", Impact},
            { "LockDuration", LockDuration},
            { "IsSkillActive", IsSkillActive},
            { "IsDashActive", IsDashActive},

            { "Duration", Duration},
            { "Robot", base.ToMap()},
        };
    }
}