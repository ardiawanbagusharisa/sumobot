using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using CoreSumo;

namespace BotAI
{
    public class AIBot_EA_MCTS : Bot
    {
        public int ReinitPerIters = 5;
        public int Iterations = 100;
        public float ActionInterval = 0.25f;
        public float SimulationTime = 0.25f;
        public float LowestScoreToReInit = -1000;
        public static List<ISumoAction> PossibleActions = new List<ISumoAction>() {
            new AccelerateAction(InputType.Script),
            new DashAction(InputType.Script),
            new SkillAction(InputType.Script),

            new TurnRightAngleAction(15f),
            new TurnRightAngleAction(45f),
            new TurnRightAngleAction(90f),
            
            new TurnLeftAngleAction(15f),
            new TurnLeftAngleAction(45f),
            new TurnLeftAngleAction(90f),
        };

        private SumoController controller;
        private SumoController enemy;
        private float decisionTimer;
        private int decisionIntervalCount = 0;
        private EA_MCTS_Node root;
        private ISumoAction actionBounceToEnemy;
        private ISumoAction actionBounceFromEnemy;

        void OnEnable()
        {
            controller = GetComponent<SumoController>();
            controller.OnPlayerBounce += OnPlayerBounce;
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
                Decide();
                if (decisionIntervalCount % ReinitPerIters == 0)
                {
                    decisionIntervalCount = 0;
                    InitNode();
                }
                decisionIntervalCount += 1;
            }
        }

        private void InitNode(ISumoAction goodAction = null, ISumoAction badAction = null)
        {
            root = null;
            root = new EA_MCTS_Node(null, null, goodAction, badAction);
            root.name = "Root";
        }

        EA_MCTS_Node Decide()
        {
            for (int i = 0; i < Iterations; i++)
            {
                EA_MCTS_Node selected = root.Select();
                float result = selected.Simulate(enemy, controller, SimulationTime);
                selected.Backpropagate(result);
            }

            EA_MCTS_Node bestChild = root.GetBestChild();

            if (bestChild.totalReward <= LowestScoreToReInit)
            {
                Debug.Log($"[AIBot_EA_MCTS] LowestScoreToReInit reached {bestChild.totalReward}");
                InitNode(badAction: bestChild.action);
                return null;
            }

            Debug.Log($"[AIBot_EA_MCTS] root actions: {string.Join(", ", root.children.Select((x) => x.name).ToList())}");
            Debug.Log($"[AIBot_EA_MCTS] selected-score: {bestChild.totalReward}, selected-action: {bestChild.action.GetType().Name}:{bestChild.action}, selected-visits: {bestChild.visits}, ");

            controller.InputProvider.EnqueueCommand(bestChild.action);
            actionBounceFromEnemy = bestChild.action;
            actionBounceToEnemy = bestChild.action;
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
                actionBounceFromEnemy = null;
            }
            else
            {
                actionBounceToEnemy = null;
            }
            controller.InputProvider.ClearCommands();
            InitNode(goodAction: actionBounceToEnemy, badAction: actionBounceFromEnemy);
        }
    }
}