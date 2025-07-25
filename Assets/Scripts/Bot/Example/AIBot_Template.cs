
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    public class AIBot_Template : Bot
    {
        public override string ID => "Template";
        public override SkillType SkillType => SkillType.Boost;

        private SumoAPI api;

        // Where the battle state changes
        public override void OnBattleStateChanged(BattleState state)
        {
        }

        // When your AI is got a collision (bounce)
        public override void OnBotCollision(EventParameter param)
        {
            PlayerSide hitter = param.Side;

            if (hitter == api.MyRobot.Side)
                Debug.Log($"My AI sent hit to enemy!");
            else
                Debug.Log($"My AI got hit!");

        }

        // Initial state of your script
        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
        }

        // OnBotUpdate() will be called everytime when the [ElapsedTime] of Battle => [Interval]
        public override void OnBotUpdate()
        {
            // If the robot is facing the enemy 90%, queue accelerate
            if (api.Angle(normalized: true) > 0.9)
            {
                Enqueue(new AccelerateAction(InputType.Script));
            }

            // To activate the queued actions
            Submit();
        }
    }
}