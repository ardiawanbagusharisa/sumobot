
using System.Collections.Generic;
using System.Linq;
using SumoCore;

namespace SumoBot.RuleBased.Fuzzy
{
    [System.Serializable]
    public abstract class FuzzyBase
    {
        public List<FuzzyRuleBase> Rules = new();
        public List<FuzzyMembershipBase> Membership = new();

        public virtual Dictionary<string, Dictionary<string, float>> Fuzzification(
               List<float> inputs)
        {
            var result = new Dictionary<string, Dictionary<string, float>>();

            for (int i = 0; i < inputs.Count; i++)
            {
                result[Membership[i].Name] = Membership[i].EvaluateMembership(inputs[i]);
            }
            return result;
        }

        public abstract List<(ISumoAction action, float score)> Defuzzification(Dictionary<string, Dictionary<string, float>> inputResult);

        public virtual List<ISumoAction> Run(List<float> inputs, int topActionsNum = 2)
        {
            var fuzzificationResult = Fuzzification(inputs);
            var defuzzificationResult = Defuzzification(fuzzificationResult);
            var topActions = defuzzificationResult
                .OrderByDescending(x => x.score)
                .Take(topActionsNum)
                .Select(x => x.action);
            return topActions.ToList();
        }
    }

    [System.Serializable]
    public abstract class FuzzyRuleBase
    {
        public Dictionary<string, string> Conditions = new();
        public ISumoAction Action;

        public abstract float GetOutputScore(
            Dictionary<string, Dictionary<string, float>> fuzzyInputs,
            out float strength);
    }

    [System.Serializable]
    public abstract class FuzzyMembershipBase
    {
        public string Name;
        public List<FuzzySetBase> Sets;

        public abstract Dictionary<string, float> EvaluateMembership(float input);
    }

    [System.Serializable]
    public abstract class FuzzySetBase
    {
        public string Label;
        public abstract float EvaluateSet(float input);
    }
}