using System.Collections.Generic;
using SumoBot.Graph;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    /// <summary>
    /// A <see cref="Bot"/> whose behavior is driven by a player-authored module graph (decision-8)
    /// instead of hardcoded C#. The graph is stored as JSON in a serialized field so Unity's
    /// Instantiate — which <see cref="BotManager.Assign"/> uses to clone a bot per side — carries
    /// it into the clone. Each tick it snapshots the world through <see cref="SumoApiSensors"/>,
    /// runs the pure <see cref="GraphInterpreter"/>, and enqueues the resulting actions.
    ///
    /// Fail-safe: a missing or invalid graph (or an interpreter error) makes the bot idle rather
    /// than throw during battle. Because the graph is a fixed set of known modules over data,
    /// there is no arbitrary code to sandbox (decision-8).
    /// </summary>
    public class GraphBot : Bot
    {
        [SerializeField] private string graphJson;
        [SerializeField] private string botId = "GraphBot";

        private GraphInterpreter interpreter;
        private SumoAPI api;
        private bool runnable;

        public override string ID => string.IsNullOrEmpty(botId) ? "GraphBot" : botId;

        /// <summary>
        /// Set the graph this bot runs. Call before handing the instance to
        /// <see cref="BotManager.AssignGraphBot"/>; the graph is stored as JSON so it survives the
        /// ScriptableObject Instantiate that Assign performs.
        /// </summary>
        public void Configure(BotGraph graph, string id = null)
        {
            graphJson = graph != null ? GraphSerializer.ToJson(graph) : null;
            if (!string.IsNullOrEmpty(id)) botId = id;
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            runnable = false;
            interpreter = null;

            BotGraph graph = null;
            if (!string.IsNullOrEmpty(graphJson))
            {
                try { graph = GraphSerializer.FromJson(graphJson); }
                catch (System.Exception e) { Debug.LogError($"[GraphBot] Failed to parse graph: {e.Message}"); }
            }

            var library = ModuleLibrary.BuildDefault();
            if (graph != null && GraphValidator.IsValid(graph, library))
            {
                interpreter = new GraphInterpreter(graph, library);
                runnable = true;
            }
            else
            {
                Debug.LogWarning("[GraphBot] Graph missing or invalid; bot will idle.");
            }
        }

        public override void OnBotUpdate()
        {
            if (!runnable || api == null) return;

            IReadOnlyList<GraphActionIntent> intents;
            try
            {
                intents = interpreter.Evaluate(new SumoApiSensors(api));
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[GraphBot] Interpreter error; idling: {e.Message}");
                runnable = false;
                return;
            }

            foreach (var intent in intents)
            {
                ISumoAction action = ToAction(intent);
                if (action != null) Enqueue(action);
            }
            Submit();
        }

        public override void OnBotCollision(BounceEvent bounceEvent) { }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner) { }

        private static ISumoAction ToAction(GraphActionIntent intent)
        {
            switch (intent.Kind)
            {
                case GraphActionKind.Accelerate: return new AccelerateAction(InputType.Script, intent.Duration);
                case GraphActionKind.TurnLeft: return new TurnAction(InputType.Script, ActionType.TurnLeft, intent.Duration);
                case GraphActionKind.TurnRight: return new TurnAction(InputType.Script, ActionType.TurnRight, intent.Duration);
                case GraphActionKind.Dash: return new DashAction(InputType.Script);
                case GraphActionKind.SkillBoost: return new SkillAction(InputType.Script, ActionType.SkillBoost);
                case GraphActionKind.SkillStone: return new SkillAction(InputType.Script, ActionType.SkillStone);
                default: return null;
            }
        }
    }
}
