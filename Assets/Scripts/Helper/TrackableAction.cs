using System;
using System.Collections.Generic;

[Serializable]
public class TrackableAction
{
    [NonSerialized]
    private Action<object[]> _action;

    [NonSerialized]
    private readonly HashSet<Delegate> _subscribers = new();

    public void Subscribe(Action<object[]> callback)
    {
        if (_subscribers.Add(callback))
        {
            _action += callback;
        }
    }

    public void Unsubscribe(Action<object[]> callback)
    {
        if (_subscribers.Remove(callback))
        {
            _action -= callback;
        }
    }

    public void Invoke(object[] param)
    {
        _action?.Invoke(param);
    }

    public void Invoke(object arg)
    {
        Invoke(new object[] { arg });
    }

    public void Invoke()
    {
        Invoke(new object[] { });
    }

    public IReadOnlyCollection<Delegate> Subscribers => _subscribers;

    public int SubscribersCount => _subscribers.Count;
    public IEnumerable<string> GetSubscriberDescriptions()
    {
        foreach (var d in _subscribers)
        {
            var method = d.Method;
            var target = d.Target;
            yield return $"{target?.GetType().Name ?? "Static"}.{method.Name}";
        }
    }
}
