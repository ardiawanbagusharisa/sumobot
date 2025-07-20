using System.Collections.Generic;
using System.Linq;
using SumoCore;
using UnityEngine;

namespace SumoBot
{
    [System.Serializable]
    public class AIBot_GA_Genome
    {
        public float[] weights = new float[AIBot_EA_GA.PossibleActions.Count];
        public float fitness;

        public AIBot_GA_Genome Clone()
        {
            AIBot_GA_Genome clone = new()
            {
                weights = (float[])weights.Clone()
            };
            return clone;
        }

        public void Mutate(float rate)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += Random.Range(-rate, rate);
            }
        }

        public static AIBot_GA_Genome Crossover(AIBot_GA_Genome a, AIBot_GA_Genome b)
        {
            var child = new AIBot_GA_Genome();
            for (int i = 0; i < child.weights.Length; i++)
            {
                child.weights[i] = Random.value < 0.5f ? a.weights[i] : b.weights[i];
            }
            return child;
        }

        public List<ISumoAction> GetBestAction(int amount = 1)
        {
            List<ISumoAction> actions = new() { };
            float bestValue = float.MinValue;

            for (int i = 0; i < weights.Length; i++)
            {
                if (weights[i] > bestValue)
                {
                    bestValue = weights[i];
                    actions.Add(AIBot_EA_GA.PossibleActions[i]);
                }
            }
            
            return actions.TakeLast(amount).ToList();
        }

    }
}