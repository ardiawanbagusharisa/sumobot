using System.Collections.Generic;
using SumoCore;
using Unity.VisualScripting;

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
    }
}