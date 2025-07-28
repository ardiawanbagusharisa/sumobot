
using System.Collections.Generic;
using SumoCore;
using SumoLog;
using SumoManager;
using UnityEngine;

namespace SumoBot.RuleBased.Fuzzy
{
    [System.Serializable]
    public class AIBot_RuleBased_Fuzzy_Config
    {
        public string Name = "Fuzzy";
        public SkillType DefaultSkillType = SkillType.Boost;

        public FuzzyBase Fuzzy = new FuzzySugeno();

    }

    class AIBot_RuleBased_Fuzzy : Bot
    {
        public AIBot_RuleBased_Fuzzy_Config config = new();
        public override string ID => config.Name;
        public override SkillType SkillType => config.DefaultSkillType;

        private SumoAPI api;


        public override void OnBotUpdate()
        {
            ClearCommands();

            float distanceEnemy = api.DistanceNormalized();
            float enemySide = api.Angle(normalized: true);
            float distanceFromArena = api.DistanceNormalized(
                targetPos: api.BattleInfo.ArenaPosition);
            float angleToEnemy = api.AngleDeg() / 360;
            float angleToArena = api.AngleDeg(targetPos: api.BattleInfo.ArenaPosition) / 360;

            List<float> inputs = new() {
                distanceEnemy,
                angleToEnemy,
                enemySide,
                distanceFromArena,
                angleToArena };

            var result = config.Fuzzy.Run(
                inputs: inputs,
                topActionsNum: 2);

            foreach (var act in result)
            {
                if (act is SkillAction)
                {
                    if (act.Type != config.DefaultSkillType.ToActionType() || api.MyRobot.Skill.IsSkillOnCooldown)
                    {
                        continue;
                    }
                }

                if (act.Type == ActionType.Dash && api.MyRobot.IsDashOnCooldown)
                {
                    continue;
                }

                if (act is TurnAction && api.IsActionActive(act))
                {
                    continue;
                }
                Enqueue(act);
            }

            Submit();
        }

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            config.Fuzzy.Rules.GenerateSugenoRule();
            config.Fuzzy.Membership.GenerateTriangular();
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
        }
    }
}