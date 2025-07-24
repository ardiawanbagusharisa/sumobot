using System.Collections.Generic;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{

    [System.Serializable]
    public class AIBot_EA_GA_Config
    {
        public int PopulationSize = 40;
        public int GenerationCount = 7;
        public float MutationRate = 0.5f;
        public int ActionsPerInterval = 2;
        public string Name = "Genetic Algorithm Example";
        public SkillType DefaultSkillType = SkillType.Boost;
    }

    public class AIBot_EA_GA : Bot
    {
        public AIBot_EA_GA_Config config = new();

        public override string ID => config.Name;
        public override SkillType SkillType => config.DefaultSkillType;

        public static List<ISumoAction> PossibleActions = new() {
            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.1f),
            new TurnAction(InputType.Script, ActionType.TurnLeft, 0.3f),

            new TurnAction(InputType.Script, ActionType.TurnRight, 0.1f),
            new TurnAction(InputType.Script, ActionType.TurnRight, 0.3f),

            new AccelerateAction(InputType.Script, 0.1f),
            new AccelerateAction(InputType.Script, 0.2f),
            new AccelerateAction(InputType.Script, 0.3f),

            new DashAction(InputType.Script),
            new SkillAction(InputType.Script),
        };

        private List<AIBot_GA_Genome> population = new();
        private AIBot_GA_Genome brain;
        private SumoAPI api;

        public override void OnBotUpdate()
        {
            brain = Run();
            List<ISumoAction> actions = brain.GetBestAction(config.ActionsPerInterval);
            actions.ForEach(action => Enqueue(action));

            Submit();
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
        }

        public override void OnBotCollision(EventParameter param)
        {
            ClearCommands();
        }

        public override void OnBattleStateChanged(BattleState state)
        {
        }

        public float Evaluate(AIBot_GA_Genome genome)
        {
            float fitness = 0f;

            List<ISumoAction> actions = genome.GetBestAction(config.ActionsPerInterval);
            foreach (var action in actions)
            {
                (Vector2, float) simResult = api.Simulate(action);
                Vector2 aiPos = simResult.Item1;
                float aiDir = simResult.Item2;

                if (api.Distance(api.BattleInfo.ArenaPosition, aiPos).magnitude > api.BattleInfo.ArenaRadius)
                    return -999f;

                if (action is DashAction)
                {
                    if (api.CanExecute(action))
                        fitness += 2;
                }

                if (action is SkillAction)
                {
                    if (api.CanExecute(action))
                        fitness += 2;
                }

                float angleScore = api.Angle(oriPos: aiPos, oriRot: aiDir, normalized: true);
                float distScore = api.DistanceNormalized(oriPos: aiPos);

                fitness += angleScore + distScore;
            }
            genome.fitness = fitness;
            return fitness;
        }



        public AIBot_GA_Genome Run()
        {
            for (int i = 0; i < config.PopulationSize; i++)
            {
                population.Add(new AIBot_GA_Genome());
            }

            for (int gen = 0; gen < config.GenerationCount; gen++)
            {
                foreach (var genome in population)
                {
                    Evaluate(genome);
                }

                population.Sort((a, b) => b.fitness.CompareTo(a.fitness));

                List<AIBot_GA_Genome> nextGen = new();

                for (int i = 0; i < config.PopulationSize / 2; i++)
                {
                    nextGen.Add(population[i].Clone());

                    var child = AIBot_GA_Genome.Crossover(
                        population[i],
                        population[Random.Range(0, config.PopulationSize / 2)]);

                    child.Mutate(config.MutationRate);
                    nextGen.Add(child);
                }

                population = nextGen;
            }

            return population[0];
        }
    }
}