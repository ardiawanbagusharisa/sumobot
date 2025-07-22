using System;
using System.Collections.Generic;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public abstract class Bot : ScriptableObject
    {
        #region Runtime properties
        
        private InputProvider provider;

        private Queue<ISumoAction> actions;

        internal void SetProvider(InputProvider provider)
        {
            actions = new Queue<ISumoAction>();
            this.provider = provider;
        }
        #endregion

        #region Abstract properties & method

        public abstract string ID { get; }

        public abstract SkillType SkillType { get; }

        public abstract void OnBotInit(PlayerSide side, SumoAPI botAPI);

        // Called when elapsed time of battle timer is satisfy with the interval
        public virtual void OnBotUpdate()
        {
            provider.EnqueueCommands(actions);
        }

        // Called when two robots get into collision (Bounce), [side] is the collider.
        public abstract void OnBotCollision(EventParameter param);

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
        #endregion
    }
}