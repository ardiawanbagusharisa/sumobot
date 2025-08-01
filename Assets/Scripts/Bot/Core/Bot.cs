using System;
using System.Collections.Generic;
using SumoCore;
using SumoInput;
using SumoLog;
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
        // [BounceEvent.Actor] is the one who made a contact
        public abstract void OnBotCollision(BounceEvent bounceEvent);

        // Called every battle state is changing
        // [state] can be one of [Preparing, Countdown, Battle_Ongoing, Battle_End, Battle_Reset]
        // [winner] will be given when the [state] is [Battle_End]
        public abstract void OnBattleStateChanged(BattleState state, BattleWinner? winner);

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

        public virtual void OnBotDestroy() { }
        #endregion
    }
}