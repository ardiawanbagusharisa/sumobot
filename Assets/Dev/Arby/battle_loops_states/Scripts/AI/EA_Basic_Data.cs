using UnityEngine;

    namespace BotAI
    {
    [System.Serializable]
    public class EA_Basic_Data
    {
        public float weightAngle;
        public float weightDistance;
        public float threshold;

        public EA_Basic_Data Clone()
        {
            return new EA_Basic_Data();
        }

        public void Mutate(float rate)
        {
            weightAngle += Random.Range(-rate, rate);
            weightDistance += Random.Range(-rate, rate);
            threshold += Random.Range(-rate, rate);
        }

        public static EA_Basic_Data Crossover(EA_Basic_Data a, EA_Basic_Data b)
        {
            return new EA_Basic_Data
            {
                weightAngle = Random.value < 0.5f ? a.weightAngle : b.weightAngle,
                weightDistance = Random.value < 0.5f ? a.weightDistance : b.weightDistance,
                threshold = Random.value < 0.5f ? a.threshold : b.threshold
            };
        }
    }
}