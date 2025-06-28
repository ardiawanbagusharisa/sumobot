using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using SumoInput;
using SumoCore;
using SumoManager;

namespace SumoBot
{
    // [CreateAssetMenu(fileName = "BOT_MCTS", menuName = "Bot/MCTS")]
    public class AIBot_EA_MCTS : Bot
    {
        public override string ID => Name;
        public override float Interval => ActionInterval;

        public string Name = "MCTS";
        public float ActionInterval = 0.1f;

        private PlayerSide side;
        private BattleState currState;

        #region AI
        private EA_MCTS_Node root;
        private List<ISumoAction> lastActionsFromEnemy;
        private List<ISumoAction> lastActionsToEnemy;
        private Queue<ISumoAction> actionsQueue = new();

        public Dictionary<string, EA_MCTS_Node> AllNodes = new();
        public float SimulationTime = 0.25f;
        public float LowestScoreToReInit = -500;
        private float decisionTimer;
        private int decisionIntervalCount = 0;
        public static List<ISumoAction> PossibleActions = new List<ISumoAction>() {
            new AccelerateAction(InputType.Script),
            new DashAction(InputType.Script),
            new SkillAction(InputType.Script),
            new TurnAction(InputType.Script, ActionType.TurnLeftWithAngle, 15f),
            new TurnAction(InputType.Script, ActionType.TurnRightWithAngle, 15f),
        };
        public int ReinitPerIters = 2;
        public int Iterations = 100;
        #endregion AI

        private BotAPI api;

        void OnBattleChanged(BattleState state)
        {
            if (state == BattleState.Battle_End)
                InitNode();
        }

        public override void OnBotUpdate()
        {
            if (currState != BattleState.Battle_Ongoing) return;

            decisionTimer += Time.deltaTime;
            if (decisionTimer >= ActionInterval)
            {
                decisionTimer = 0f;
                if (decisionIntervalCount % ReinitPerIters == 0)
                {
                    decisionIntervalCount = 0;
                    InitNode();
                }
                Decide();
                decisionIntervalCount += 1;
            }

            DeQueueWhenAvailable();

            base.OnBotUpdate();
        }

        public override void OnBattleStateChanged(BattleState state)
        {
            currState = state;
            OnBattleChanged(state);
        }

        public override void OnBotCollision(object[] args)
        {
            if (side == (PlayerSide)args[0])
                lastActionsFromEnemy = null;
            else
                lastActionsToEnemy = null;
            InitNode();
        }

        public override void OnBotInit(PlayerSide side, BotAPI botAPI)
        {
            api = botAPI;
            InitNode();
        }


        #region Custom Functions

        void DeQueueWhenAvailable()
        {
            while (!api.Controller.IsMovementDisabled && actionsQueue.Count > 0)
            {
                Enqueue(actionsQueue.Dequeue());
            }
        }

        private void InitNode()
        {
            Debug.Log($"AllNodes {AllNodes.Count()}");
            AllNodes.Clear();
            root = new EA_MCTS_Node(
                null,
                new List<ISumoAction>(PossibleActions),
                goodAction: lastActionsToEnemy,
                badAction: lastActionsFromEnemy);
            root.Side = side;
            root.ID = "Root";
            root.Init(AllNodes);
        }

        EA_MCTS_Node Decide()
        {
            for (int i = 0; i < Iterations; i++)
            {
                EA_MCTS_Node selected = root.Select();
                var expanded = selected.Expand(AllNodes);
                if (expanded != null)
                {
                    var result = expanded.Simulate(api, SimulationTime);
                    expanded.Backpropagate(result);
                }

            }

            EA_MCTS_Node bestChild = root.GetBestChild();
            if (bestChild == null)
                return null;

            if (bestChild.totalReward <= LowestScoreToReInit)
            {
                Debug.Log($"[AIBot_EA_MCTS] LowestScoreToReInit reached {bestChild.totalReward}");
                InitNode();
                return null;
            }

            Debug.Log($"[AIBot_EA_MCTS] selected-score: {bestChild.totalReward}, selected-action(s): {bestChild.ID} selected-visits: {bestChild.visits}, ");

            lastActionsToEnemy = bestChild.actions;

            foreach (var act in bestChild.actions)
            {
                act.Reason = bestChild.GetHighestScoreType().ToString();
                actionsQueue.Enqueue(act);
            }
            return bestChild;
        }

        #endregion
    }
}