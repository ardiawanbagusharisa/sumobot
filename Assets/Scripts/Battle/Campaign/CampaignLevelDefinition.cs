using SumoInput;
using SumoManager;
using UnityEngine;

namespace SumoCampaign
{
    public enum CampaignTask
    {
        Move, TurnAndApproach, AlignedDash, EdgeRecovery, RingOut,
        CommandMove, CommandSequence, CommandDash, CommandRecovery, CommandRingOut,
        FsmApproaching, FsmAttacking, FsmDodging, FsmRecovering, FsmSequence
    }

    [CreateAssetMenu(menuName = "Sumobot/Campaign Level")]
    public sealed class CampaignLevelDefinition : ScriptableObject
    {
        public string Id;
        public string Title;
        public string Scene;
        public InputType Input;
        public string PlayerBot;
        public string OpponentBot;
        public float OpponentInterval;
        public float BattleSeconds;
        public RoundSystem Rounds;
        public int FirstWinCoins;
        [TextArea] public string Instructions;
        public float ArenaRadius = 10f;
        [Range(.1f, .8f)] public float SpawnFraction = .35f;
        [Range(-.8f, .8f)] public float OpponentSpawnFraction = .35f;
        public CampaignTask Task;
        [TextArea] public string Question;
        public string[] Answers;
        public int CorrectAnswer;
    }
}
