using System.Collections.Generic;
using System.Linq;
using SumoCore;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.SceneManagement;
using SumoInput;

namespace SumoManager
{
    public enum GuideTab
    {
        Gameplay,
        Rules,
        Control,

    }
    public class BattleUIManager : MonoBehaviour
    {
        public static BattleUIManager Instance { get; private set; }

        #region UI Elements properties
        [Header("Main Panels")]
        public List<GameObject> BattlePanels = new();

        [Header("Pre-battle UI")]
        public TMP_Dropdown LeftSkill;
        public TMP_Dropdown RightSkill;

        [Header("Battle UI")]
        public GameObject PausePanel;
        public TMP_Text BattleStateUI;
        public TMP_Text Countdown;
        public TMP_Text RoundSystem;
        public TMP_Text Round;
        public TMP_Text Timer;
        public List<SumoCostume> PlayerHUD = new();
        public ChartManager PostBattleChart;
        public TMP_Text PostBattleChartTitle;


        [Header("Battle UI - Left Player")]
        public TMP_Text LeftScore;
        public TMP_Text LeftFinalScore;
        public CooldownUIGroupSet LeftSkillUI;
        public CooldownUIGroupSet LeftDashUI;

        // [Todo] Temporary
        public GameObject LeftDashBuff;
        public GameObject LeftSkillBuff;


        [Header("Battle UI - Right Player")]
        public TMP_Text RightScore;
        public TMP_Text RightFinalScore;
        public CooldownUIGroupSet RightSkillUI;
        public CooldownUIGroupSet RightDashUI;

        // [Todo] Temporary
        public GameObject RightDashBuff;
        public GameObject RightSkillBuff;

        [Header("Battle UI - Guide Menu")]
        public GameObject GuidePanel;
        public TMP_Text GuideContent;
        public Button GuideGameplayTab;
        public Button GuideRulesTab;
        public Button GuideControlTab;
        public ScrollRect GuideScrollRect;
        public Color GuideActiveTabColor = new Color(1f, 0.89f, 0.62f);
        public Color GuideInactiveTabColor = new Color(0.88f, 0.88f, 0.88f);

        #endregion


        #region Runtime (readonly) Properties
        private readonly string gameplayContent = @"
Sumobot is a robot game that pits strength in a sumo ring. Your
main goal is to push your opponent out of the ring, while staying in the circle.

1. How to Play
- Control your sumo robot to move, dodge, and attack your opponent.
- Use special moves like Dash to make quick attacks or avoid enemy attacks.
- There are two types of special skills, Boost (for quick movements) and Stone (to freeze and withstand collisions).
- Each round starts with both robots in their respective positions.
- The game ends when one of the robots exits the ring.
- The player who remains in the ring is declared the winner.

2. Playing Tips
- Use skills and Dash at the right time to avoid collisions or make surprise attacks.
- Pay attention to your opponent''s position and direction of movement, look for gaps to attack from the side or behind.
- Avoid being too close to the edge of the ring so that you don''t get pushed out easily.
";

        private readonly string rulesContent = @"
1. Win and Lose
- Players will win if they successfully push their opponent out of the arena (sumo ring).
- Players will win if they are able to survive in the arena until time runs out.
- Players are declared the loser if their robot is pushed out first.
- Players are declared the winner if they are able to win the most rounds

2. Prohibitions
You are not allowed to exploit bugs to gain an advantage.

3. Round System
- The game can take place in several rounds (according to the settings).
- Rounds consists three types, which are Best of one, three, and five.
- The time for one round is 30-60 seconds
- The winning score will be calculated after each round is completed.

4. Skills and Cooldown
- Player can only choose Boost or Stone.
- Players must wait for the cooldown to finish before they can use the skill again.

5. Penalties
If a player violates the rules (eg: bug exploit), the round can be disqualified.
";

        private readonly string controlsContent = @"
W / O - Forward
A / K - Turn Left
D / ; - Turn Right
C / M - Special Skill
Left Shift / Right Shift - Dash
";
        private Dictionary<GuideTab, Button> guideButtonsMap => new()
        {
            {GuideTab.Gameplay,GuideGameplayTab},
            {GuideTab.Rules,GuideRulesTab},
            {GuideTab.Control,GuideControlTab},
        };
        private Dictionary<GuideTab, string> guideContentsMap => new()
        {
            {GuideTab.Gameplay,gameplayContent},
            {GuideTab.Rules,rulesContent},
            {GuideTab.Control,controlsContent},
        };

        #endregion


        #region Unity methods
        private void Awake()
        {
            if (Instance != null)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;

        }

        private void OnEnable()
        {
            BattleManager.Instance.Actions[BattleManager.OnBattleChanged].Subscribe(OnBattleChanged);
        }

        private void OnDisable()
        {
            BattleManager.Instance.Actions[BattleManager.OnBattleChanged].Unsubscribe(OnBattleChanged);
        }

        private void FixedUpdate()
        {
            BattleManager battle = BattleManager.Instance;
            if (battle.CurrentState == BattleState.Battle_Ongoing ||
            battle.CurrentState == BattleState.Battle_End ||
            battle.CurrentState == BattleState.Battle_Reset)
            {
                Timer.SetText(Mathf.CeilToInt(battle.TimeLeft).ToString());
                SumoController leftPlayer = battle.Battle.LeftPlayer;
                SumoController rightPlayer = battle.Battle.RightPlayer;

                UpdateActionUI(leftPlayer, LeftSkillUI, LeftDashUI);
                UpdateActionUI(rightPlayer, RightSkillUI, RightDashUI);

                LeftDashBuff.SetActive(leftPlayer.IsDashActive);
                LeftSkillBuff.SetActive(leftPlayer.Skill.IsActive);
                RightDashBuff.SetActive(rightPlayer.IsDashActive);
                RightSkillBuff.SetActive(rightPlayer.Skill.IsActive);
            }
            else
            {
                ResetActionUI(LeftSkillUI, LeftDashUI);
                ResetActionUI(RightSkillUI, RightDashUI);

                // Reset timer UI
                if (Timer != null)
                    Timer.SetText(battle.BattleTime.ToString());
            }
        }
        #endregion

        #region Battle changes
        private void OnBattleChanged(ActionParameter param)
        {
            var battle = param.Battle;
            RoundSystem.SetText($"Best of {(int)battle.RoundSystem}");
            Round.SetText($"Round {battle.CurrentRound?.RoundNumber}");

            Round round = battle.CurrentRound;
            BattleState state = BattleManager.Instance.CurrentState;
            BattleStateUI.SetText(state.ToString());

            SumoController leftPlayer = battle.LeftPlayer;
            SumoController rightPlayer = battle.RightPlayer;
            switch (state)
            {
                case BattleState.PreBatle_Preparing:
                    GuidePanel?.SetActive(false);
                    PausePanel?.SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(true);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    LeftSkill.value = (int)leftPlayer.Skill.Type;
                    RightSkill.value = (int)rightPlayer.Skill.Type;
                    LeftFinalScore.SetText("");
                    RightFinalScore.SetText("");
                    PlayerHUD.ForEach((x) =>
                    {
                        if (x.Side == Placement.Left)
                            x.AttachToHUD(leftPlayer.Costume);
                        else if (x.Side == Placement.Right)
                            x.AttachToHUD(rightPlayer.Costume);
                    });
                    break;
                case BattleState.Battle_Preparing:
                    battle.LeftPlayer.Actions[SumoController.OnSkillAssigned].Subscribe(OnSkillAssigned);
                    battle.RightPlayer.Actions[SumoController.OnSkillAssigned].Subscribe(OnSkillAssigned);

                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(true);
                    InitActionUI(LeftSkillUI, LeftDashUI);
                    InitActionUI(RightSkillUI, RightDashUI);
                    ClearScore();
                    LeftSkillUI.SetText(leftPlayer.Skill.Type.ToString());
                    RightSkillUI.SetText(rightPlayer.Skill.Type.ToString());
                    Countdown.SetText("");
                    RoundSystem.SetText("");
                    Round.SetText("");
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    break;
                case BattleState.Battle_Countdown:
                    BattleManager.Instance.Actions[BattleManager.OnCountdownChanged].Subscribe(OnCountdownChanged);
                    break;
                case BattleState.Battle_Ongoing:
                    Countdown.SetText("");
                    BattleManager.Instance.Actions[BattleManager.OnCountdownChanged].Unsubscribe(OnCountdownChanged);
                    break;
                case BattleState.Battle_End:
                    leftPlayer.Actions[SumoController.OnSkillAssigned].Unsubscribe(OnSkillAssigned);
                    rightPlayer.Actions[SumoController.OnSkillAssigned].Unsubscribe(OnSkillAssigned);
                    break;
                case BattleState.PostBattle_ShowResult:
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(true);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);

                    LeftFinalScore.SetText(battle.LeftWinCount.ToString());
                    RightFinalScore.SetText(battle.RightWinCount.ToString());

                    SumoCostume winnerHUD = PlayerHUD.FirstOrDefault((x) => x.Side == Placement.Winner);
                    if (winnerHUD)
                    {
                        SumoController winner = battle.GetBattleWinner().GetRobotWinner(battle);
                        winnerHUD?.AttachToHUD(winner.Costume);
                    }

                    ShowActionChart(battle);
                    break;
            }
            UpdateScore(battle);
        }

        private void ShowActionChart(Battle _)
        {
            Debug.Log("ShowGameSummary Called");
            for (int gameIdx = 0; gameIdx < LogManager.Log.Games.Count; gameIdx++)
            {
                LogManager.GameLog gameLog = LogManager.Log.Games[gameIdx];

                for (int roundIdx = 0; roundIdx < gameLog.Rounds.Count; roundIdx++)
                {
                    Dictionary<int, (int, int)> actionTakenCountMap = new();

                    LogManager.RoundLog roundLog = gameLog.Rounds[roundIdx];

                    float avgTimeFrame = roundLog.Duration / 12;

                    int timeRange = 1;

                    for (float i = 0; i < roundLog.Duration; i += avgTimeFrame)
                    {
                        int leftActionTaken = roundLog.PlayerEvents.Where((x) => x.UpdatedAt < (i + avgTimeFrame) && x.UpdatedAt >= i && x.Category == "Action" && x.Actor == "Left" && x.IsStart).Count();

                        int rightActionTaken = roundLog.PlayerEvents.Where((x) => x.UpdatedAt < (i + avgTimeFrame) && x.UpdatedAt > i && x.Category == "Action" && x.Actor == "Right" && x.IsStart).Count();

                        actionTakenCountMap[timeRange] = (leftActionTaken, rightActionTaken);

                        timeRange += 1;
                    }

                    var leftC = actionTakenCountMap.Select((x) => (float)x.Value.Item1).ToArray();
                    var rightC = actionTakenCountMap.Select((x) => (float)x.Value.Item2).ToArray();


                    ChartSeries chartLeft = new ChartSeries($"Left_Round_{roundIdx + 1}", actionTakenCountMap.Select((x) => (float)x.Value.Item1).ToArray(), ChartSeries.ChartType.Bar, Color.green);

                    ChartSeries chartRight = new ChartSeries($"Right_Round_{roundIdx + 1}", actionTakenCountMap.Select((x) => (float)x.Value.Item2).ToArray(), ChartSeries.ChartType.Bar, Color.red);

                    PostBattleChart.Setup(xGridSpacing: (int)avgTimeFrame);
                    PostBattleChart.AddChartSeries(chartLeft);
                    PostBattleChart.AddChartSeries(chartRight);
                    PostBattleChart.DrawChart();
                }
            }

            // PostBattleChart.AddChartSeries
        }

        private void OnCountdownChanged(ActionParameter param)
        {
            Countdown.SetText(param.Float.ToString());
        }

        private void OnSkillAssigned(ActionParameter param)
        {
            if (param.Side == PlayerSide.Left)
                LeftSkillUI.SetText(param.SkillType.ToString());
            else
                RightSkillUI.SetText(param.SkillType.ToString());
        }

        private void InitActionUI(CooldownUIGroupSet skill, CooldownUIGroupSet dash)
        {
            InputType inputType = BattleManager.Instance.BattleInputType;

            skill.SetVisible(inputType);
            dash.SetVisible(inputType);
        }

        private void UpdateActionUI(SumoController player, CooldownUIGroupSet skill, CooldownUIGroupSet dash)
        {
            InputType inputType = BattleManager.Instance.BattleInputType;

            skill.SetCooldown(player.Skill.CooldownNormalized, inputType);
            dash.SetCooldown(player.DashCooldownNormalized, inputType);
        }

        private void ResetActionUI(CooldownUIGroupSet skill, CooldownUIGroupSet dash)
        {
            InputType inputType = BattleManager.Instance.BattleInputType;

            skill.Reset(inputType);
            dash.Reset(inputType);
        }

        private void UpdateScore(Battle battleInfo)
        {
            if (battleInfo.Winners.Count() == 0)
            {
                ClearScore();
                return;
            }

            LeftScore.SetText(battleInfo.LeftWinCount.ToString());
            RightScore.SetText(battleInfo.RightWinCount.ToString());
        }

        private void ClearScore()
        {
            LeftScore.SetText("0");
            RightScore.SetText("0");
        }
        #endregion

        #region Battle menu
        public void Pause()
        {
            PausePanel.SetActive(true);
            Time.timeScale = 0;
        }

        public void Resume()
        {
            PausePanel.SetActive(false);
            Time.timeScale = 1;
        }

        public void Restart()
        {
            Time.timeScale = 1;
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }

        public void ShowGuide()
        {
            if (GuidePanel != null)
            {
                GuidePanel.SetActive(true);
                ShowGameplayGuide();
            }
        }

        public void HideGuide()
        {
            if (GuidePanel != null)
                GuidePanel.SetActive(false);
        }

        public void ShowGameplayGuide() => ShowGuide(GuideTab.Gameplay);
        public void ShowRulesGuide() => ShowGuide(GuideTab.Rules);
        public void ShowControlGuide() => ShowGuide(GuideTab.Control);

        private void ShowGuide(GuideTab tab)
        {
            GuideContent.text = guideContentsMap[tab].Trim();

            foreach (var button in guideButtonsMap)
            {
                bool isActive = tab == button.Key;

                ColorBlock colors = button.Value.colors;
                colors.normalColor = isActive ? GuideActiveTabColor : GuideInactiveTabColor;
                colors.selectedColor = isActive ? GuideActiveTabColor : GuideInactiveTabColor;
                button.Value.colors = colors;

                if (button.Value.TryGetComponent(out Image tabImg))
                    tabImg.color = isActive ? GuideActiveTabColor : GuideInactiveTabColor;
            }

            if (GuideScrollRect != null)
                StartCoroutine(ResetScrollGuide());
        }

        private IEnumerator ResetScrollGuide()
        {
            yield return null;
            GuideScrollRect.verticalNormalizedPosition = 1f;
        }
        #endregion
    }
}
