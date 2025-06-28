using System.Collections.Generic;
using CoreSumo;
using Unity.VisualScripting;

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
}