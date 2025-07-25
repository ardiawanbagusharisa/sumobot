using System;
using System.Collections.Generic;

namespace SumoHelper
{
    [Serializable]
    public class EventRegistry
    {
        private Dictionary<string, TrackableEvent> events = new();

        public TrackableEvent this[string key]
        {
            get
            {
                if (!events.ContainsKey(key))
                    events[key] = new TrackableEvent();
                return events[key];
            }
        }

        public IReadOnlyDictionary<string, TrackableEvent> Events => events;
    }
}