using System;
using System.Collections.Generic;
using SumoCore;
using SumoLog;
using SumoManager;
using UnityEngine;

namespace SumoHelper
{
    [Serializable]
    public class TrackableEvent
    {
        [NonSerialized]
        private Action<EventParameter> action;

        [NonSerialized]
        private readonly HashSet<Delegate> subscribers = new();

        public void Subscribe(Action<EventParameter> callback)
        {
            if (subscribers.Add(callback))
                action += callback;

        }

        public void Unsubscribe(Action<EventParameter> callback)
        {
            if (subscribers.Remove(callback))
                action -= callback;
        }

        public void Invoke(EventParameter param)
        {
            action?.Invoke(param);
        }

        public void Invoke()
        {
            Invoke(new EventParameter());
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
public class EventParameter
{
    public readonly ISumoAction Action;
    public readonly PlayerSide Side;
    public readonly bool Bool;
    public readonly float Float;
    public readonly BounceEvent BounceEvent;
    public readonly BattleState BattleState;
    public BattleWinner? Winner;
    public readonly SkillType SkillType;

    public EventParameter(
        ISumoAction actionParam = null,
        PlayerSide? sideParam = null,
        bool? boolParam = null,
        float? floatParam = null,
        BattleState? battleStateParam = null,
        BounceEvent bounceInfoParam = null,
        BattleWinner? winnerParam = null,
        SkillType? skillType = null)
    {

        if (sideParam != null)
            Side = (PlayerSide)sideParam;
        if (skillType != null)
            SkillType = (SkillType)skillType;
        if (battleStateParam != null)
            BattleState = (BattleState)battleStateParam;
        if (sideParam != null)
            Action = actionParam;
        if (boolParam != null)
            Bool = (bool)boolParam;
        if (floatParam != null)
            Float = (float)floatParam;
        if (bounceInfoParam != null)
            BounceEvent = bounceInfoParam;
        if (winnerParam != null)
            Winner = (BattleWinner)winnerParam;
    }
}

[Serializable]
public class BounceEvent
{
    public PlayerSide Actor;
    public CollisionLog MyInfo;
    public CollisionLog EnemyInfo;

    public BounceEvent(
        PlayerSide actor,
        CollisionLog actorInfo,
        CollisionLog targetInfo
    )
    {
        Actor = actor;
        MyInfo = actorInfo;
        EnemyInfo = targetInfo;
    }

    public override string ToString()
    {
        return $"BounceEvent -> Hitter: {Actor}, \nActor -> {MyInfo.ToMap()}\nTarget -> {EnemyInfo.ToMap()}";
    }
}