
using SumoCore;
using SumoInput;
using SumoManager;

namespace SumoBot
{
    public class AIBot_Template : Bot
    {
        public override string ID => "Dummy";

        // Ranged between 0.1f to 10f
        public override float Interval => 0.1f;

        public override SkillType SkillType => SkillType.Boost;

        // Where the battle state changes
        public override void OnBattleStateChanged(BattleState state)
        {
            throw new System.NotImplementedException();
        }

        // When your AI is got a collision (bounce)
        public override void OnBotCollision(ActionParameter param)
        {
            throw new System.NotImplementedException();
        }

        // Initial state of your script
        public override void OnBotInit(PlayerSide side, SumoAPI botAPI)
        {
            throw new System.NotImplementedException();
        }

        // OnBotUpdate() will be called everytime when the [ElapsedTime] of Battle => [Interval]
        public override void OnBotUpdate()
        {
            Enqueue(new AccelerateAction(InputType.Script));
            Enqueue(new AccelerateAction(InputType.Script));
            Enqueue(new AccelerateAction(InputType.Script));
            Enqueue(new AccelerateAction(InputType.Script));

            base.OnBotUpdate(); // Call this to activate your commands
        }
    }
}