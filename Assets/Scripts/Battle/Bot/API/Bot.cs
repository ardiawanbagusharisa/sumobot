using System;
using System.Collections.Generic;
using CoreSumo;
using UnityEngine;


public abstract class Bot : ScriptableObject
{
    // [Todo]: Inherited Bot script shouldn't be able to access internal attributes
    // Need to configure asmdef (assembly scope)
    internal float BotElapsed = 0;
    internal InputProvider provider;
    internal Queue<ISumoAction> actions;
    internal void SetProvider(InputProvider provider)
    {
        actions = new Queue<ISumoAction>();
        this.provider = provider;
    }

    public abstract string ID { get; }

    [Range(0.1f, 10f)]
    public abstract float Interval { get; }

    public abstract void OnBotInit(PlayerSide side, BotAPI botAPI);

    // Called when elapsed time of battle timer is satisfy with the interval
    public virtual void OnBotUpdate()
    {
        provider.EnqueueCommands(actions);
    }

    // Called when two robots get into collision (Bounce), [side] is the collider.
    public abstract void OnBotCollision(PlayerSide side);

    // Called whenever battle state ischanged
    public abstract void OnBattleStateChanged(BattleState state);

    // Actions will be dequeued / invoked when the interval is set
    public virtual void Enqueue(ISumoAction action)
    {
        actions.Enqueue(action);
    }

    public virtual void ClearCommands()
    {
        actions.Clear();
    }
}