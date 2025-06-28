using System;
using System.Collections.Generic;

[Serializable]
public class ActionRegistry
{
    private Dictionary<string, TrackableAction> _actions = new();

    public TrackableAction this[string key]
    {
        get
        {
            if (!_actions.ContainsKey(key))
                _actions[key] = new TrackableAction();
            return _actions[key];
        }
    }

    public IReadOnlyDictionary<string, TrackableAction> Actions => _actions;
}
