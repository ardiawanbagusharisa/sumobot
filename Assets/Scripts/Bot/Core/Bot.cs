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

        // Called once Bot initialized (BattleState.preparing)
        public abstract void OnBotInit(SumoAPI botAPI);

        // Called every battle tick is satisfied.
        public abstract void OnBotUpdate();

        // Called every two robots get a collision (Bounce).
        // [param.Side] -> hitter.
        // [param.Float] -> own robot lock duration.
        public abstract void OnBotCollision(EventParameter param);

        // Called every battle state is changing
        public abstract void OnBattleStateChanged(BattleState state);

        // Add one action to local queue
        public virtual void Enqueue(ISumoAction action)
        {
            actions.Enqueue(action);
        }

        // Send your local queue to game's queue
        // actions that already sent will be executed in order every battle tick
        public void Submit()
        {
            provider.EnqueueCommand(actions);
        }

        // Clear your queue locally.
        public virtual void ClearCommands()
        {
            actions.Clear();
        }
        #endregion
    }
}