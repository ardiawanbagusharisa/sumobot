using System;
using System.Collections.Generic;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    // For test / debug, switch between ScriptableObject / Component
    public abstract class Bot : MonoBehaviour
    {
        // [Todo]: Inherited Bot script shouldn't be able to access internal attributes
        // Need to configure asmdef (assembly scope)
        internal float ElapsedTime = 0;
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

        public abstract void OnBotInit(PlayerSide side, SumoAPI botAPI);

        // Called when elapsed time of battle timer is satisfy with the interval
        public virtual void OnBotUpdate()
        {
            provider.EnqueueCommands(actions);
        }

        // Called when two robots get into collision (Bounce), [side] is the collider.
        public abstract void OnBotCollision(object[] side);

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
}