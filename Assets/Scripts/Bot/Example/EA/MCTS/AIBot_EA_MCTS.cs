using UnityEngine;
using System.Collections.Generic;
using SumoInput;
using SumoCore;
using SumoManager;
using System;

namespace SumoBot.EA.MCTS
{

    [Serializable]
    public class AI_MCTS_Config
    {
        public int ResetIterationAt = 2;
        public float Iterations = 100;
        public float UCBConstant = 1.41f;
        public float ScoreLimit = -300;
        public string Name = "MCTS Example";
        public SkillType DefaultSkillType = SkillType.Boost;
    }

    public class AIBot_EA_MCTS : Bot
    {
        public AI_MCTS_Config config = new();
        public override string ID => config.Name;
        public override SkillType SkillType => config.DefaultSkillType;


        #region Runtime properties
        private EA_MCTS_Node root;
        private List<ISumoAction> lastActionsFromEnemy;
        private List<ISumoAction> lastActionsToEnemy;
        public Dictionary<string, EA_MCTS_Node> AllNodes = new();
        public static List<ISumoAction> PossibleActions = new() {
            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.1f),
            new TurnAction(InputType.Script, ActionType.TurnRight, 0.1f),

            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.3f),
            new TurnAction(InputType.Script, ActionType.TurnRight, 0.3f),

            new AccelerateAction(InputType.Script, 0.1f),
            new AccelerateAction(InputType.Script, 0.3f),
            new DashAction(InputType.Script),
            new SkillAction(InputType.Script),
        };
        private BattleState currState;
        private PlayerSide side;
        private int elapsedInterval = 0;
        #endregion

        private SumoAPI api;
        public override void OnBotUpdate()
        {

            elapsedInterval += 1;

            if (elapsedInterval >= config.ResetIterationAt)
            {
                elapsedInterval = 0;
                InitNode();
            }
            Decide();

            Submit();
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
            currState = state;

            if (currState == BattleState.Battle_Countdown)
            {
                lastActionsFromEnemy = null;
                lastActionsToEnemy = null;
                InitNode();
            }
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
            if (side == bounceEvent.Actor)
                lastActionsFromEnemy = null;
            else
                lastActionsToEnemy = null;
            ClearCommands();
            InitNode();
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            InitNode();
        }

        #region Custom Functions
        private void InitNode()
        {
            AllNodes.Clear();
            root = new EA_MCTS_Node(
                null,
                new List<ISumoAction>(PossibleActions),
                goodAction: lastActionsToEnemy,
                badAction: lastActionsFromEnemy)
            {
                Side = side,
                ID = "Root"
            };
            root.Init(AllNodes);
        }

        EA_MCTS_Node Decide()
        {
            for (int i = 0; i < config.Iterations; i++)
            {
                EA_MCTS_Node selected = root.Select(config);
                var expanded = selected.Expand(AllNodes);
                if (expanded != null)
                {
                    var result = expanded.Simulate(api, config);
                    expanded.Backpropagate(result);
                }

            }
            EA_MCTS_Node bestChild = root.GetBestChild();
            if (bestChild == null)
                return null;

            if (bestChild.totalReward <= config.ScoreLimit)
            {
                Debug.Log($"[AIBot_EA_MCTS] LowestScoreToReInit reached {bestChild.totalReward}");
                InitNode();
                return null;
            }

            // Debug.Log($"[AIBot_EA_MCTS] selected-score: {bestChild.totalReward}, selected-action(s): {bestChild.ID} selected-visits: {bestChild.visits}, {string.Join("->", bestChild.actions.Select((x) => x.FullName))}");

            lastActionsToEnemy = bestChild.actions;

            foreach (var act in bestChild.actions)
            {
                act.Reason = bestChild.GetHighestScoreType().ToString();
                // if (act is TurnAction && api.IsActionActive(act))
                //     continue;
                Enqueue(act);
            }
            return bestChild;
        }

        #endregion
    }
}