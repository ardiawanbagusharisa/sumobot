using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public class AIBot_EA_GA : Bot
    {
        public override string ID => Name;
        public override float Interval => ActionInterval;
        public override SkillType SkillType => SkillType.Boost;

        public string Name => "GA";

        public float EvaluationInterval = 2f;
        public float ActionInterval = 0.5f;

        [HideInInspector]
        public EA_GA_Data brain;
        [HideInInspector]
        public EA_GA_Data memory;

        private float fitness;
        private float evaluationTimer;

        private SumoAPI api;
        private BattleState currState;

        void EvaluateFitness()
        {
            if (memory == null || fitness > CalculateFitness(memory))
                memory = brain.Clone();
            else
                brain = EA_GA_Data.Crossover(brain, memory);
            brain.Mutate(0.1f);

            fitness = 0f;
        }

        float CalculateFitness(EA_GA_Data data)
        {
            return api.DistanceNormalized();
        }

        void RandomizeBrain(EA_GA_Data data)
        {
            data.weightAngle = Random.Range(-1f, 1f);
            data.weightDistance = Random.Range(-1f, 1f);
            data.threshold = Random.Range(-1f, 1f);

            Debug.Log($"[AIBot_EA_Basic][RandomizeBrain] weightAngle: {brain.weightAngle}, weightDistance: {brain.weightAngle}");
        }

        public void Decide()
        {
            float value = brain.weightDistance * api.DistanceNormalized();

            Debug.Log($"[AIBot_EA_Basic] value: {value}, weightAngle: {brain.weightAngle}, normalizedAngle: {api.Angle(normalized: true)}, weightDistance: {brain.weightAngle}, normalizedDistance: {api.DistanceNormalized()}");

            float accelDuration = 0.3f;
            float signedAngleToEnemy = api.Angle();

            if (Mathf.Abs(signedAngleToEnemy) < 20f)
            {
                if (Mathf.Abs(value) > 0.6f)
                    Enqueue(new SkillAction(InputType.Script));
                else if (Mathf.Abs(value) > 0.3f)
                    Enqueue(new DashAction(InputType.Script));
                else
                    Enqueue(new AccelerateAction(InputType.Script, accelDuration));
            }
            else
            {

                if (signedAngleToEnemy < 0)
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnLeft, ActionInterval));
                else
                    Enqueue(new TurnAction(InputType.Script, ActionType.TurnRight, ActionInterval));
            }

            fitness += 1f; // Example: reward for taking an action
        }
        public override void OnBotInit(PlayerSide side, SumoAPI botAPI)
        {
            api = botAPI;
            brain = new EA_GA_Data();
            RandomizeBrain(brain);
        }

        public override void OnBotUpdate()
        {
            if (evaluationTimer >= EvaluationInterval)
            {
                EvaluateFitness();
                evaluationTimer = 0f;
            }

            Decide();

            base.OnBotUpdate();
        }

        public override void OnBotCollision(ActionParameter param)
        {
            ClearCommands();
        }

        public override void OnBattleStateChanged(BattleState state)
        {
            currState = state;
        }
    }
}