using System.Collections.Generic;
using UnityEngine;

namespace SumoBot.RuleBased.Fuzzy
{

    [System.Serializable]
    public class FuzzyTriangleValue : FuzzySetBase
    {
        public FuzzyTriangleValue(string label, float center, float spread)
        {
            Label = label;
            Center = center;
            Spread = spread;
        }
        public float Center;
        public float Spread;

        public override float EvaluateSet(float input)
        {
            return Mathf.Clamp01(1f - Mathf.Abs(input - Center) / Spread);
        }
    }

    [System.Serializable]
    public class FuzzyTriangleSet : FuzzyMembershipBase
    {
        public FuzzyTriangleSet(string name, List<FuzzySetBase> values)
        {
            Name = name;
            Sets = values;
        }

        public override Dictionary<string, float> EvaluateMembership(float input)
        {
            var result = new Dictionary<string, float>();
            foreach (var cat in Sets)
            {
                result[cat.Label] = cat.EvaluateSet(input);
            }
            return result;
        }
    }

    public static class FuzzyTriangleSetExtensions
    {
        public static void GenerateTriangular(this List<FuzzyMembershipBase> value)
        {
            value.AddRange(new List<FuzzyMembershipBase>
            {
                new FuzzyTriangleSet("distance_enemy", new()
                    {
                        new FuzzyTriangleValue ( label : "far", center : 0f, spread : 0.3f ),
                        new FuzzyTriangleValue ( label : "medium", center : 0.6f, spread : 0.25f ),
                        new FuzzyTriangleValue ( label : "close", center : 1f, spread : 0.6f ),
                    }),
                new FuzzyTriangleSet("angle_enemy", new()
                    {
                        new FuzzyTriangleValue(label: "front", center: 0.0f, spread: 0.125f),
                        new FuzzyTriangleValue(label: "front_left", center: 0.125f, spread: 0.125f),
                        new FuzzyTriangleValue(label: "left", center: 0.25f, spread: 0.125f),
                        new FuzzyTriangleValue(label: "back_left", center: 0.375f, spread: 0.125f),
                        new FuzzyTriangleValue(label: "back", center: 0.5f, spread: 0.125f),
                        new FuzzyTriangleValue(label: "back_right", center: 0.625f, spread: 0.125f),
                        new FuzzyTriangleValue(label: "right", center: 0.75f, spread: 0.125f),
                        new FuzzyTriangleValue(label: "front_right", center: 0.875f, spread: 0.125f),
                    }),
                new FuzzyTriangleSet("side_enemy", new()
                    {
                        new FuzzyTriangleValue(label: "behind", center: 0f, spread: 0.4f),
                        new FuzzyTriangleValue(label: "side", center: 0.7f, spread: 0.3f),
                        new FuzzyTriangleValue(label: "front", center: 1f, spread: 0.7f),
                    }),
                new FuzzyTriangleSet("distance_arena", new()
                    {
                        new FuzzyTriangleValue(label: "near_border", center: 0f, spread: 0.3f),
                        new FuzzyTriangleValue(label: "close_border", center: 0.6f, spread: 0.2f),
                        new FuzzyTriangleValue(label: "center", center: 1f, spread: 0.8f),
                    }),
                new FuzzyTriangleSet("angle_arena", new()
                    {
                        new FuzzyTriangleValue(label: "front", center: 0f, spread: 0.16f),
                        new FuzzyTriangleValue(label: "behind", center: 0.5f, spread: 0.16f),
                    }),
            });
        }
    }
}