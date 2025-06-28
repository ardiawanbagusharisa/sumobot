using System;
using System.Collections.Generic;

[Serializable]
public class ActionRegistry
{
    private Dictionary<string, TrackableAction> actions = new();

    public TrackableAction this[string key]
    {
        get
        {
            if (!actions.ContainsKey(key))
                actions[key] = new TrackableAction();
            return actions[key];
        }
    }

    public IReadOnlyDictionary<string, TrackableAction> Actions => actions;
}
