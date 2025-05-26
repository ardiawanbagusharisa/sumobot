using UnityEngine;
using CoreSumoRobot;
using BattleLoop;
using System.Collections.Generic;
using System.Linq;

namespace BotAI
{
    public class AIBot_EA_MCTS : MonoBehaviour
    {
        private SumoRobotController controller;
        private SumoRobotController enemy;

        public float DecisionActionInterval = 0.25f;
        public int DecisionResetInterval = 5;
        private float decisionTimer;
        private int decisionIntervalCount = 0;

        EA_MCTS_Node root;
        public static Dictionary<string, ISumoAction> PossibleActions = new Dictionary<string, ISumoAction>() {
            {"AccelerateAction",new AccelerateAction(InputType.Script)},
            {"DashAction",new DashAction(InputType.Script)},
            {"TurnRightAction",new TurnRightAngleAction(45f)},
            {"TurnLeftAction",new TurnLeftAngleAction(45f)},
            {"SkillAction",new SkillAction(InputType.Script)},
        };

        void OnEnable()
        {
            controller = GetComponent<SumoRobotController>();
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
            if (decisionTimer >= DecisionActionInterval)
            {
                decisionTimer = 0f;
                if (decisionIntervalCount % DecisionResetInterval == 0)
                {
                    InitNode();
                }
                Decide();
                decisionIntervalCount += 1;
            }
        }

        private void InitNode()
        {
            root = new EA_MCTS_Node(null, null);
            root.name = "Root";
        }

        void Decide()
        {
            for (int i = 0; i < 100; i++)
            {
                EA_MCTS_Node selected = root.Select();
                float result = selected.Simulate(enemy, controller, DecisionActionInterval);
                selected.Backpropagate(result);
            }

            EA_MCTS_Node bestChild = root.GetBestChild();
            var childActions = new List<string>();

            Debug.Log($"[AIBot_EA_MCTS] root actions: {string.Join(", ", root.children.Select((x) => x.name).ToList())}");
            Debug.Log($"[AIBot_EA_MCTS] score: {bestChild.totalReward}, action: {bestChild.action}, visits: {bestChild.visits}, ");

            controller.InputProvider.EnqueueCommand(PossibleActions[bestChild.action]);
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
            controller.InputProvider.ClearCommands();
            InitNode();
        }
    }
}