using System;
using System.Collections.Generic;
using SumoCore;
using SumoManager;

namespace SumoHelper
{
    [Serializable]
    public class TrackableAction
    {
        [NonSerialized]
        private Action<ActionParameter> action;

        [NonSerialized]
        private readonly HashSet<Delegate> subscribers = new();

        public void Subscribe(Action<ActionParameter> callback)
        {
            if (subscribers.Add(callback))
                action += callback;

        }

        public void Unsubscribe(Action<ActionParameter> callback)
        {
            if (subscribers.Remove(callback))
                action -= callback;
        }

        public void Invoke(ActionParameter param)
        {
            action?.Invoke(param);
        }

        public void Invoke()
        {
            Invoke(new ActionParameter());
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


[Serializable]
public class ActionParameter
{
    public ISumoAction Action;
    public PlayerSide Side;
    public bool Bool;
    public float Float;
    public Battle Battle;
    public SkillType SkillType;

    public ActionParameter(
        ISumoAction actionParam = null,
        PlayerSide? sideParam = null,
        bool? boolParam = null,
        float? floatParam = null,
        Battle battleParam = null,
        SkillType? skillType = null)
    {

        if (sideParam != null)
            Side = (PlayerSide)sideParam;
        if (skillType != null)
            SkillType = (SkillType)skillType;
        if (sideParam != null)
            Action = actionParam;
        if (boolParam != null)
            Bool = (bool)boolParam;
        if (floatParam != null)
            Float = (float)floatParam;
        if (battleParam != null)
            Battle = battleParam;
    }
}