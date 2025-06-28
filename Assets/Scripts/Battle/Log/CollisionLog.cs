
using System.Collections.Generic;


public class CollisionLog : BaseLog
{
    public bool IsActor;
    public float Impact;
    public float BounceResistance;
    public float LockDuration = float.NaN;
    public bool IsSkillActive;

    public float Duration;

    public new Dictionary<string, dynamic> ToMap()
    {
        var data = new Dictionary<string, dynamic>()
        {
            { "IsActor", IsActor},
            { "Impact", Impact},
            { "LockDuration", LockDuration},
            { "IsSkillActive", IsSkillActive},
            
            { "Duration", Duration},
            { "Robot", base.ToMap()},
        };
        return data;
    }
}