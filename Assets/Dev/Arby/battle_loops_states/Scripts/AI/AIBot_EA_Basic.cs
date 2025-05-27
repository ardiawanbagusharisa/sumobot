using BattleLoop;
using CoreSumoRobot;
using UnityEngine;

namespace BotAI
{
    public class AIBot_EA_Basic : MonoBehaviour
    {
        public float EvaluationInterval = 5f;
        public float ActionInterval = 0.4f;

        [HideInInspector]
        public EA_Basic_Data brain;
        [HideInInspector]
        public EA_Basic_Data memory;

        private SumoRobotController controller;
        private SumoRobotController enemy;
        private float fitness;
        private float evaluationTimer;
        private float actionTimer = 0f;

        void OnEnable()
        {
            controller = GetComponent<SumoRobotController>();
            controller.OnPlayerBounce += OnPlayerBounce;
        }

        void Start()
        {
            brain = new EA_Basic_Data();
            RandomizeBrain(brain);
        }

        void OnDisable()
        {
            controller.OnPlayerBounce -= OnPlayerBounce;
        }

        void Update()
        {
            if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
            {
                evaluationTimer += Time.deltaTime;
                if (evaluationTimer >= EvaluationInterval)
                {
                    EvaluateFitness();
                    evaluationTimer = 0f;
                }

                actionTimer += Time.deltaTime;
                if (actionTimer >= ActionInterval)
                {
                    actionTimer = 0f;
                    Decide();
                }
            }
        }

        void FixedUpdate()
        {
            if (enemy == null)
            {
                enemy = controller.Side == PlayerSide.Left
                    ? BattleManager.Instance.Battle.RightPlayer
                    : BattleManager.Instance.Battle.LeftPlayer;
            }
        }

        void EvaluateFitness()
        {
            if (memory == null || fitness > CalculateFitness(memory))
            {
                memory = brain.Clone();
            }
            else
            {
                brain = EA_Basic_Data.Crossover(brain, memory);
                brain.Mutate(0.1f);
            }

            fitness = 0f;
        }

        float CalculateFitness(EA_Basic_Data data)
        {
            // Example: distance to enemy
            float dist = Vector3.Distance(transform.position, enemy.transform.position);
            return 1f / (dist + 0.01f);
        }

        void RandomizeBrain(EA_Basic_Data data)
        {
            data.weightAngle = Random.Range(-1f, 1f);
            data.weightDistance = Random.Range(-1f, 1f);
            data.threshold = Random.Range(-1f, 1f);

            Debug.Log($"[AIBot_EA_Basic][RandomizeBrain] weightAngle: {brain.weightAngle}, weightDistance: {brain.weightAngle}");
        }

        public void Decide()
        {
            if (enemy == null) return;

            Vector2 toEnemy = enemy.transform.position - transform.position;
            float angleToTarget = Vector2.SignedAngle(transform.up, toEnemy.normalized);
            float normalizedAngle = 1f - Mathf.Abs(angleToTarget) / 180f;
            float normalizedDistance = 1f - Mathf.Abs(toEnemy.magnitude) / 7f;

            float value = brain.weightDistance * normalizedDistance;

            Debug.Log($"[AIBot_EA_Basic] value: {value}, weightAngle: {brain.weightAngle}, normalizedAngle: {normalizedAngle}, weightDistance: {brain.weightAngle}, normalizedDistance: {normalizedDistance}");

            float accelDuration = 0.3f;

            if (Mathf.Abs(angleToTarget) < 20f)
            {
                if (Mathf.Abs(value) > 0.6f)
                {
                    controller.InputProvider.EnqueueCommand(new SkillAction(InputType.Script));
                }
                else if (Mathf.Abs(value) > 0.3f)
                {
                    controller.InputProvider.EnqueueCommand(new DashAction(InputType.Script));
                }
                else
                {
                    controller.InputProvider.EnqueueCommand(new AccelerateTimeAction(accelDuration));
                }
            }
            else
            {
                controller.InputProvider.EnqueueCommand(new TurnAngleAction(angleToTarget));
            }



            fitness += 1f; // Example: reward for taking an action
        }

        private void OnPlayerBounce(PlayerSide side)
        {
            controller.InputProvider.ClearCommands();
        }
    }
}