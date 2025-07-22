using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using SumoInput;
using SumoCore;
using SumoManager;
using System;

namespace SumoBot
{

    [Serializable]
    public class AI_MCTS_Config
    {
        public float ScriptInterval = 0.1f;
        public float SimulationTime = 0.3f;
        public int ResetIterationAt = 10;
        public float Iterations = 100;
        public float ScoreLimit = -300;
        public string Name = "MCTS Example";
        public SkillType DefaultSkillType = SkillType.Stone;
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
        private Queue<ISumoAction> actionsQueue = new();

        public Dictionary<string, EA_MCTS_Node> AllNodes = new();
        public static List<ISumoAction> PossibleActions = new() {
            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.1f),
            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.3f),

            new TurnAction(InputType.Script, ActionType.TurnRight, 0.1f),
            new TurnAction(InputType.Script, ActionType.TurnRight, 0.3f),

            new AccelerateAction(InputType.Script, 0.1f),

            new DashAction(InputType.Script),
            new SkillAction(InputType.Script),
        };
        private BattleState currState;
        private PlayerSide side;
        private int elapsedInterval = 0;
        #endregion

        private SumoAPI api;

        void OnBattleChanged(BattleState state)
        {
            if (state == BattleState.Battle_End)
                InitNode();
        }

        public override void OnBotUpdate()
        {
            if (elapsedInterval % config.ResetIterationAt == 0)
            {
                InitNode();
            }
            Decide();
            elapsedInterval += 1;

            DeQueueWhenAvailable();

            base.OnBotUpdate();
        }

        public override void OnBattleStateChanged(BattleState state)
        {
            currState = state;
            OnBattleChanged(state);
        }

        public override void OnBotCollision(EventParameter param)
        {
            if (side == param.Side)
                lastActionsFromEnemy = null;
            else
                lastActionsToEnemy = null;
            InitNode();
        }

        public override void OnBotInit(PlayerSide side, SumoAPI botAPI)
        {
            api = botAPI;

            // Set all duration for action is similar
            // for (int i = 0; i < PossibleActions.Count; i++)
            // {
            //     PossibleActions[i].Duration = config.SimulationTime;
            // }

            InitNode();
        }


        #region Custom Functions

        void DeQueueWhenAvailable()
        {
            while (actionsQueue.Count > 0)
            {
                Enqueue(actionsQueue.Dequeue());
            }
        }

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
                EA_MCTS_Node selected = root.Select();
                var expanded = selected.Expand(AllNodes);
                if (expanded != null)
                {
                    expanded.SortAction();
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

            Debug.Log($"[AIBot_EA_MCTS] selected-score: {bestChild.totalReward}, selected-action(s): {bestChild.ID} selected-visits: {bestChild.visits}, {string.Join("->", bestChild.actions.Select((x) => x.FullName))}");

            lastActionsToEnemy = bestChild.actions;

            foreach (var act in bestChild.actions)
            {
                // if (api.ActionLockTime(act.Type) > 0)
                // {
                //     continue;
                // }
                act.Reason = bestChild.GetHighestScoreType().ToString();
                actionsQueue.Enqueue(act);
            }
            return bestChild;
        }

        #endregion
    }
}