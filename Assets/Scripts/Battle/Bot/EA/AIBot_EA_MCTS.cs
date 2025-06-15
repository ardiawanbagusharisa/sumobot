using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using CoreSumo;

namespace BotAI
{
    public class AIBot_EA_MCTS : Bot
    {
        public int ReinitPerIters = 2;
        public int Iterations = 100;
        public float ActionInterval = 0.03f;
        public float SimulationTime = 0.3f;
        public float LowestScoreToReInit = -500;
        public static List<ISumoAction> PossibleActions = new List<ISumoAction>() {
            new AccelerateAction(InputType.Script),
            new DashAction(InputType.Script),
            new SkillAction(InputType.Script),

            new TurnRightAngleAction(15f),
            // new TurnRightAngleAction(2f),
            // new TurnRightAngleAction(45f),
            // new TurnRightAngleAction(90f),

            new TurnLeftAngleAction(15f),
            // new TurnLeftAngleAction(2f),
            // new TurnLeftAngleAction(45f),
            // new TurnLeftAngleAction(90f),
        };

        public Dictionary<string, EA_MCTS_Node> AllNodes = new();

        private SumoController controller;
        private SumoController enemy;
        private float decisionTimer;
        private int decisionIntervalCount = 0;
        private EA_MCTS_Node root;

        private List<ISumoAction> lastActionsFromEnemy;
        private List<ISumoAction> lastActionsToEnemy;

        private Queue<ISumoAction> actionsQueue = new();

        void DeQueueWhenAvailable()
        {
            while (!controller.IsMovementLocked && actionsQueue.Count > 0)
            {
                controller.InputProvider.EnqueueCommand(actionsQueue.Dequeue());
            }
        }

        void OnEnable()
        {
            controller = GetComponent<SumoController>();
            controller.OnPlayerBounce += OnPlayerBounce;
            // AllNodes.Add(controller.Side, new());
            actionsQueue = new();
            InitNode();
        }

        void Start()
        {
            BattleManager.Instance.OnBattleChanged += OnBattleChanged;
        }

        void OnDisable()
        {
            controller.OnPlayerBounce -= OnPlayerBounce;
            BattleManager.Instance.OnBattleChanged -= OnBattleChanged;
        }

        void Update()
        {
            if (enemy == null)
            {
                enemy = controller.Side == PlayerSide.Left
                    ? BattleManager.Instance.Battle.RightPlayer
                    : BattleManager.Instance.Battle.LeftPlayer;
            }

            if (BattleManager.Instance.CurrentState != BattleState.Battle_Ongoing)
                return;

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
            root.Side = controller.Side;
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
                    float result = expanded.Simulate(enemy, controller, SimulationTime);
                    expanded.Backpropagate(result);
                }

            }

            EA_MCTS_Node bestChild = root.GetBestChild();
            if (bestChild == null)
            {
                return null;
            }

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
                actionsQueue.Enqueue(act);
            }
            return bestChild;
        }

        void OnBattleChanged(Battle battle)
        {
            if (BattleManager.Instance.CurrentState == BattleState.Battle_End)
            {
                InitNode();
            }
        }

        void OnPlayerBounce(PlayerSide side)
        {
            if (side == controller.Side)
            {
                lastActionsFromEnemy = null;
            }
            else
            {
                lastActionsToEnemy = null;
            }

            controller.InputProvider.ClearCommands();
            InitNode();
        }
    }
}