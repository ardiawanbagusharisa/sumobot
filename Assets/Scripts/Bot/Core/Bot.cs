using SumoCore;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    /// <summary>
    /// Abstract base class for all bot logic in the SumoBot system.
    /// Provides a standardized interface for bot behavior during battle, including initialization,
    /// updates, collision handling, and state changes.
    /// </summary>
    public abstract class Bot : ScriptableObject
    {
        #region Runtime properties (private)

        private BotHandler handler;

        internal void Init(BotHandler handler)
        {
            this.handler = handler;
        }
        #endregion

        #region Abstract properties & method (public)

        /// <summary>
        /// Unique identifier for the bot (e.g., "Bot_01", "Stone_02"). Can be name of your Bot
        /// </summary>
        public abstract string ID { get; }

        /// <summary>
        /// Indicates whether the bot uses asynchronous operations (e.g., for ML inference).
        /// </summary>
        public virtual bool UseAsync { get; } = false;

        /// <summary>
        /// Type of skill the bot uses (e.g., Boost, Stone).
        /// </summary>
        public virtual SkillType DefaultSkillType { get; }

        /// <summary>
        /// Called once when the bot is initialized, during battle preparation.
        /// Allows bot-specific setup or initialization logic.
        /// </summary>
        public abstract void OnBotInit(SumoAPI botAPI);

        /// <summary>
        /// Called every battle tick (default 100ms), allowing the bot to update its state and decisions.
        /// Should contain AI logic for movement, skill usage, and action prediction.
        /// </summary>
        public abstract void OnBotUpdate();

        /// <summary>
        /// Called when two bots collide (bounce). The [bounceEvent] contains details about the collision.
        /// </summary>
        /// <param name="bounceEvent">Event data containing the colliding bots and collision details.</param>
        public abstract void OnBotCollision(BounceEvent bounceEvent);

        /// <summary>
        /// Called whenever the battle state changes (e.g., from Battle_Countdown to Battle_Ongoing).
        /// Provides the bot with knowledge of the current battle phase.
        /// </summary>
        /// <param name="state">Current battle state (e.g., Preparing, Battle_Ongoing).</param>
        /// <param name="winner">Optional winner if the state is Battle_End.</param>
        public abstract void OnBattleStateChanged(BattleState state, BattleWinner? winner);

        /// <summary>
        /// Adds a single action (e.g., dash, boost) to the bot's local action queue.
        /// </summary>
        /// <param name="action">The action to enqueue.</param>
        public virtual void Enqueue(ISumoAction action)
        {
            handler.Enqueue(action);
        }

        /// <summary>
        /// Transfers all queued actions from the local bot queue to the game's action queue.
        /// Actions are executed in order during each battle tick.
        /// Can only be called inside [OnBotUpdate], otherwise will throw Exception
        /// </summary>
        public void Submit()
        {
            handler.Submit();
        }

        /// <summary>
        /// Clears all actions from the local bot queue.
        /// Use when a bot wants to reset its action list (e.g., after a skill or cooldown).
        /// </summary>
        public virtual void ClearCommands()
        {
            handler.Actions.Clear();
        }

        /// <summary>
        /// Called when the bot is destroyed (e.g., game over or bot removed).
        /// Allows cleanup or finalization of resources.
        /// </summary>
        public virtual void OnBotDestroy() { }
        #endregion
    }
}