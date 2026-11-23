using System;
using System.Collections.Generic;
using SumoInput;
using SumoManager;
using UnityEngine.SceneManagement;

namespace SumoCampaign
{
    /// <summary>Fifteen authored configurations using the three existing battle templates.</summary>
    public static class CampaignCatalog
    {
        public sealed class Level
        {
            public readonly string Id;
            public readonly string Title;
            public readonly string Scene;
            public readonly InputType Input;
            public readonly string PlayerBot;
            public readonly string OpponentBot;
            public readonly float OpponentInterval;
            public readonly float BattleSeconds;
            public readonly RoundSystem Rounds;
            public readonly int FirstWinCoins;
            public readonly string Instructions;

            public Level(string id, string title, string scene, InputType input,
                string playerBot, string opponentBot, float opponentInterval,
                float battleSeconds, RoundSystem rounds, int firstWinCoins, string instructions)
            {
                Id = id;
                Title = title;
                Scene = scene;
                Input = input;
                PlayerBot = playerBot;
                OpponentBot = opponentBot;
                OpponentInterval = opponentInterval;
                BattleSeconds = battleSeconds;
                Rounds = rounds;
                FirstWinCoins = firstWinCoins;
                Instructions = instructions;
            }
        }

        private const string Buttons = "Campaign_1_1";
        private const string Commands = "Campaign_2_1";
        private const string Scripts = "Campaign_3_1";
        private const string Primitive = "AIBot_Primitive";
        private const string Fsm = "AlBot_FSM";
        private const string Tree = "AIBot_BT";
        private const string Utility = "AIBot_RuleBased_Utility";

        public static readonly IReadOnlyList<Level> Levels = new[]
        {
            new Level("1-1", "First Push", Buttons, InputType.UI, null, Primitive, 1.2f, 60f, RoundSystem.BestOf1, 30,
                "Use the on-screen buttons to turn toward the opponent and push it out of the arena."),
            new Level("1-2", "Turn and Approach", Buttons, InputType.UI, null, Primitive, .9f, 60f, RoundSystem.BestOf1, 35,
                "Practice turning before accelerating. Win one round against the faster opponent."),
            new Level("1-3", "Dash Timing", Buttons, InputType.UI, null, Primitive, .7f, 55f, RoundSystem.BestOf1, 40,
                "Line up the opponent, then use Dash at close range to finish the push."),
            new Level("1-4", "Defend the Edge", Buttons, InputType.UI, null, Fsm, .7f, 55f, RoundSystem.BestOf1, 45,
                "The opponent now tracks your position. Turn inward when you approach the arena edge."),
            new Level("1-5", "Button Challenge", Buttons, InputType.UI, null, Tree, .5f, 60f, RoundSystem.BestOf3, 55,
                "Win two rounds against a behavior-tree opponent using buttons and your equipped skill."),

            new Level("2-1", "First Command", Commands, InputType.LiveCommand, null, Primitive, 2.5f, 65f, RoundSystem.BestOf1, 35,
                "Enter live movement commands to turn, accelerate, and push the slow opponent out."),
            new Level("2-2", "Command Sequence", Commands, InputType.LiveCommand, null, Primitive, 1.8f, 65f, RoundSystem.BestOf1, 40,
                "Combine turns and acceleration in the command terminal to control your approach."),
            new Level("2-3", "Aim and Dash", Commands, InputType.LiveCommand, null, Fsm, 1.2f, 65f, RoundSystem.BestOf1, 45,
                "Use a live dash command only after your bot is facing the opponent."),
            new Level("2-4", "React Quickly", Commands, InputType.LiveCommand, null, Tree, .9f, 70f, RoundSystem.BestOf1, 50,
                "Adapt your command sequence to a behavior-tree opponent."),
            new Level("2-5", "Command Challenge", Commands, InputType.LiveCommand, null, Utility, .75f, 75f, RoundSystem.BestOf3, 60,
                "Win two rounds using live commands against a utility-based opponent."),

            new Level("3-1", "Watch the FSM", Scripts, InputType.Script, Fsm, Primitive, 1.4f, 60f, RoundSystem.BestOf1, 35,
                "Watch your FSM bot search, approach, and attack. Observe how it reacts to distance and angle."),
            new Level("3-2", "FSM Duel", Scripts, InputType.Script, Fsm, Fsm, 1f, 65f, RoundSystem.BestOf1, 40,
                "Both bots use state machines. Observe when their behavior changes."),
            new Level("3-3", "Against a Tree", Scripts, InputType.Script, Fsm, Tree, .8f, 65f, RoundSystem.BestOf1, 45,
                "Compare the FSM bot with a behavior-tree opponent."),
            new Level("3-4", "Utility Rival", Scripts, InputType.Script, Fsm, Utility, .65f, 70f, RoundSystem.BestOf1, 50,
                "Observe how a utility-based bot chooses actions against the FSM."),
            new Level("3-5", "AI Challenge", Scripts, InputType.Script, Fsm, Tree, .4f, 75f, RoundSystem.BestOf3, 65,
                "Win two rounds with the FSM bot against a fast behavior-tree opponent.")
        };

        public static Level Selected { get; private set; }

        public static void Launch(Level level)
        {
            if (level == null) throw new ArgumentNullException(nameof(level));
            Selected = level;
            SceneManager.LoadScene(level.Scene);
        }

        public static void ClearSelection() => Selected = null;
    }
}
