
using System;

namespace SumoHelper
{
    [Serializable]
    public class MinMax
    {
        public float min;
        public float max;

        public MinMax(float min, float max)
        {
            this.min = min;
            this.max = max;
        }
    }
}