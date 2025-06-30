using System;
using System.Collections.Generic;

namespace SumoHelper
{
    [Serializable]
    public class TrackableAction
    {
        [NonSerialized]
        private Action<object[]> action;

        [NonSerialized]
        private readonly HashSet<Delegate> subscribers = new();

        public void Subscribe(Action<object[]> callback)
        {
            if (subscribers.Add(callback))
                action += callback;
            
        }

        public void Unsubscribe(Action<object[]> callback)
        {
            if (subscribers.Remove(callback))
                action -= callback;
        }

        public void Invoke(object[] param)
        {
            action?.Invoke(param);
        }

        public void Invoke(object arg)
        {
            Invoke(new object[] { arg });
        }

        public void Invoke()
        {
            Invoke(new object[] { });
        }

        public IReadOnlyCollection<Delegate> Subscribers => subscribers;

        public int SubscribersCount => subscribers.Count;
        public IEnumerable<string> GetSubscriberDescriptions()
        {
            foreach (var d in subscribers)
            {
                var method = d.Method;
                var target = d.Target;
                yield return $"{target?.GetType().Name ?? "Static"}.{method.Name}";
            }
        }
    }
}