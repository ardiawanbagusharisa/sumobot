
using SumoCore;
using SumoInput;
using SumoManager;

namespace SumoBot
{
    public class AIBot_Template : Bot
    {
        public override string ID => "Template";

        public override SkillType SkillType => SkillType.Boost;

        // Where the battle state changes
        public override void OnBattleStateChanged(BattleState state)
        {
        }

        // When your AI is got a collision (bounce)
        public override void OnBotCollision(EventParameter param)
        {
        }

        // Initial state of your script
        public override void OnBotInit(PlayerSide side, SumoAPI botAPI)
        {
        }

        // OnBotUpdate() will be called everytime when the [ElapsedTime] of Battle => [Interval]
        public override void OnBotUpdate()
        {
            // Enqueue(new AccelerateAction(InputType.Script));
            // Enqueue(new AccelerateAction(InputType.Script));
            // Enqueue(new AccelerateAction(InputType.Script));
            // Enqueue(new AccelerateAction(InputType.Script));

            base.OnBotUpdate(); // Call this to activate your commands
        }
    }
}