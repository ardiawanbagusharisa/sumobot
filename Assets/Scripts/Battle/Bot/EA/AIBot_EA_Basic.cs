using CoreSumo;
using UnityEngine;

namespace BotAI
{
    // [CreateAssetMenu(fileName = "BOT_GA", menuName = "Bot/GA")]
    public class AIBot_EA_Basic : Bot
    {
        public override string ID => Name;
        public override float Interval => ActionInterval;

        public string Name => "GA";
        public float EvaluationInterval = 5f;
        public float ActionInterval = 0.4f;

        [HideInInspector]
        public EA_Basic_Data brain;
        [HideInInspector]
        public EA_Basic_Data memory;

        private float fitness;
        private float evaluationTimer;
        private float actionTimer = 0f;

        private BotAPI api;
        private BattleState currState;

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
            float dist = Vector3.Distance(api.MyTransform.position, api.EnemyTransform.position);
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
            Vector2 toEnemy = api.EnemyTransform.position - api.MyTransform.position;
            float angleToTarget = Vector2.SignedAngle(api.MyTransform.up, toEnemy.normalized);
            float normalizedAngle = 1f - Mathf.Abs(angleToTarget) / 180f;
            float normalizedDistance = 1f - Mathf.Abs(toEnemy.magnitude) / 7f;

            float value = brain.weightDistance * normalizedDistance;

            Debug.Log($"[AIBot_EA_Basic] value: {value}, weightAngle: {brain.weightAngle}, normalizedAngle: {normalizedAngle}, weightDistance: {brain.weightAngle}, normalizedDistance: {normalizedDistance}");

            float accelDuration = 0.3f;

            if (Mathf.Abs(angleToTarget) < 20f)
            {
                if (Mathf.Abs(value) > 0.6f)
                {
                    Enqueue(new SkillAction(InputType.Script));
                }
                else if (Mathf.Abs(value) > 0.3f)
                {
                    Enqueue(new DashAction(InputType.Script));
                }
                else
                {
                    Enqueue(new AccelerateAction(InputType.Script, accelDuration));
                }
            }
            else
            {

                if (angleToTarget < 0)
                {
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeftWithAngle, Mathf.Abs(angleToTarget)));
                }
                else
                {
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRightWithAngle, Mathf.Abs(angleToTarget)));
                }
            }

            fitness += 1f; // Example: reward for taking an action
        }

        private void OnPlayerBounce(PlayerSide side)
        {
            ClearCommands();
        }

        public override void OnBotInit(PlayerSide side, BotAPI botAPI)
        {
            api = botAPI;
            brain = new EA_Basic_Data();
            RandomizeBrain(brain);
        }

        public override void OnBotUpdate()
        {
            if (currState != BattleState.Battle_Ongoing) return;

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

            base.OnBotUpdate();
        }

        public override void OnBotCollision(object[] args)
        {
            OnPlayerBounce((PlayerSide)args[0]);
        }

        public override void OnBattleStateChanged(BattleState state)
        {
            currState = state;
        }
    }
}