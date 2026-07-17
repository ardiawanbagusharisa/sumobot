using SumoBot;
using SumoInput;
using SumoManager;

namespace SumoLeaderboard
{
    /// <summary>
    /// Builds a BattleResultRecord from the live battle objects.
    /// Human sides are identified by their persistent PlayerProfile; script-driven
    /// sides are ranked as bot entities ("bot:&lt;Bot.ID&gt;") so the same bot
    /// accumulates one rating regardless of which side it played on.
    /// </summary>
    public static class LeaderboardRecordFactory
    {
        public const string HumanBotPlaceholder = "-";
        public const string BotProfilePrefix = "bot:";

        public static BattleResultRecord FromBattle(BattleManager battleManager)
        {
            InputType leftType = battleManager.LeftInputType;
            InputType rightType = battleManager.RightInputType;

            Bot leftBot = battleManager.BotManager != null ? battleManager.BotManager.Left : null;
            Bot rightBot = battleManager.BotManager != null ? battleManager.BotManager.Right : null;

            (string leftId, string leftName, string leftBotName) =
                SideIdentity(leftType, leftBot, GameManager.Instance.Left);
            (string rightId, string rightName, string rightBotName) =
                SideIdentity(rightType, rightBot, GameManager.Instance.Right);

            return new BattleResultRecord
            {
                // The Battle scene is offline multiplayer today. When campaign
                // mode ships, tag its matches GameMode.Campaign here.
                GameMode = GameMode.Multiplayer,
                Mode = LeaderboardMapping.DerivePlayerMode(leftType, rightType),
                LeftControl = leftType.ToControlCategory(),
                RightControl = rightType.ToControlCategory(),
                LeftProfileID = leftId,
                RightProfileID = rightId,
                LeftName = leftName,
                RightName = rightName,
                LeftBot = leftBotName,
                RightBot = rightBotName,
                Winner = battleManager.Battle.GetBattleWinner() ?? BattleWinner.Draw,
                RoundSystem = battleManager.RoundSystem,
                LeftRoundWins = battleManager.Battle.LeftWinCount,
                RightRoundWins = battleManager.Battle.RightWinCount,
            };
        }

        private static (string id, string name, string bot) SideIdentity(
            InputType type, Bot bot, PlayerProfile profile)
        {
            if (type.IsAI() && bot != null)
                return (BotProfilePrefix + bot.ID, bot.ID, bot.ID);

            if (profile != null)
                return (