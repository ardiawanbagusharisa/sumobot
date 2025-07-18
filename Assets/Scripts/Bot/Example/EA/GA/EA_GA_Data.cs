using UnityEngine;

namespace SumoBot
{
    public class EA_GA_Data
    {
        public float weightAngle;
        public float weightDistance;
        public float threshold;

        public EA_GA_Data Clone()
        {
            return new EA_GA_Data();
        }

        public void Mutate(float rate)
        {
            weightAngle += Random.Range(-rate, rate);
            weightDistance += Random.Range(-rate, rate);
            threshold += Random.Range(-rate, rate);
        }

        public static EA_GA_Data Crossover(EA_GA_Data a, EA_GA_Data b)
        {
            return new EA_GA_Data
            {
                weightAngle = Random.value < 0.5f ? a.weightAngle : b.weightAngle,
                weightDistance = Random.value < 0.5f ? a.weightDistance : b.weightDistance,
                threshold = Random.value < 0.5f ? a.threshold : b.threshold
            };
        }
    }
}