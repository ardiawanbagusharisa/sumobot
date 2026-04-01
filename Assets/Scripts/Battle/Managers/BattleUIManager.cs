using System.Collections.Generic;
using System.Linq;
using SumoCore;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.SceneManagement;
using SumoInput;
using SumoBot;

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

        [Header("Battle UI")]
        public GameObject PausePanel;
        public TMP_Text BattleStateUI;
        public TMP_Text Countdown;
        public TMP_Text RoundSystem;
        public TMP_Text Round;
        public TMP_Text Timer;
        public List<SumoCostume> PlayerHUD = new();
        public Sprite CrownSprite;

        [Header("Battle UI - Left Player")]
        public TMP_Dropdown LeftInputType;
        public TMP_Dropdown LeftSkill;
        public TMP_Dropdown LeftScript;
        public Button LeftCustomBtn;
        public GameObject LeftCrown;
        public TMP_Text LeftScore;
        public TMP_Text LeftFinalScore;
        public CooldownUIGroupSet LeftSkillUI;
        public CooldownUIGroupSet LeftDashUI;

        public GameObject LeftDashBuff;
        public GameObject LeftSkillBuff;


        [Header("Battle UI - Right Player")]
        public TMP_Dropdown RightInputType;
        public TMP_Dropdown RightSkill;
        public TMP_Dropdown RightScript;
        public Button RightCustomBtn;
        public GameObject RightCrown;
        public TMP_Text RightScore;
        public TMP_Text RightFinalScore;
        public CooldownUIGroupSet RightSkillUI;
        public CooldownUIGroupSet RightDashUI;

        // [Todo] Temporary
        public GameObject RightDashBuff;
        public GameObject RightSkillBuff;

        [Header("Battle UI - Guide Menu")]
        public GameObject GuidePanel;
        public GameObject ButtonGuide;
        public GameObject ButtonMainMenu;
        public GameObject ButtonResume;
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
Sumobot: sumo robot battle arena game. 
Goal: push the opponent out of the ring arena. 

1. How to Play
- Control your bot using: buttons, live commands, AI script, or visual script.
- Available actions: accelerate, turn left, turn right, dash, and special skill.
- Skill types: Boost (quick movements) and Stone (freeze & reflect collisions).
- Keep your bot survive until the timer runs out, or push the opponent out. 

2. Tips
- Timing your actions to avoid collisions or make surprise attacks.
- Pay attention to positions and directions; attack from the blind spot.
- Avoid being too close to the ring's edge.
";

		private readonly string rulesContent = @"
1. Round System
- A game takes place in rounds (1, 3, or 5), and a duration (30s or 60s). 

2. Skills and Cooldown
- Can only choose 1 skill: Boost or Stone, each has a cooldown. 

3. Win and Lose 
- Bot wins a battle round immediatley if they push the opponent out first.
- The game winning score will be calculated after all round is completed. 

4. Report 
If you found a bug or vioaltion, report to us to make this game better. 
";

		private readonly string controlsContent = @"
1. Control - Buttons

		     Left Player	    	    Right Player 
Forward		W			O
Turn Left		A			K
Turn Right		D			;
Dash			L. Shift			R. Shift 
Skill			C			M 

2. Control - Live Commands
- Type ""help"" to show all commands. 
- Press `tab` for autocomplete, press `enter` to execute a command.

3. Control - AI Script (for advance player only)
- Download and modify the AI script templates in our GitHub page, then submit it. 

4. Control - Visual Script 
*Coming soon*
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
            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleChanged);
        }

        private void OnDisable()
        {
            BattleManager.Instance.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleChanged);
        }

        void Start()
        {
            LeftInputType.onValueChanged.AddListener((v) =>
            {
                SetInputMode(PlayerSide.Left, v);
            });
            RightInputType.onValueChanged.AddListener((v) =>
            {
                SetInputMode(PlayerSide.Right, v);
            });
            LeftSkill.onValueChanged.AddListener((v) =>
            {
                SetDefaultSkill(PlayerSide.Left, v);
            });
            RightSkill.onValueChanged.AddListener((v) =>
            {
                SetDefaultSkill(PlayerSide.Right, v);
            });
            LeftScript.onValueChanged.AddListener((v) =>
            {
                OnBotSelect(PlayerSide.Left, v);
            });
            RightScript.onValueChanged.AddListener((v) =>
            {
                OnBotSelect(PlayerSide.Right, v);
            });

        }

        private void FixedUpdate()
        {
            BattleManager battle = BattleManager.Instance;
            if (battle.CurrentState <= BattleState.Battle_Preparing) return;

            SumoController leftPlayer = battle.Battle.LeftPlayer;
            SumoController rightPlayer = battle.Battle.RightPlayer;

            if (battle.CurrentState == BattleState.Battle_Ongoing ||
            battle.CurrentState == BattleState.Battle_End ||
            battle.CurrentState == BattleState.Battle_Reset)
            {
                Timer.SetText(Mathf.CeilToInt(battle.TimeLeft).ToString());

                UpdateActionUI(LeftSkillUI, LeftDashUI, leftPlayer);
                UpdateActionUI(RightSkillUI, RightDashUI, rightPlayer);

                LeftDashBuff.SetActive(leftPlayer.IsDashActive);
                LeftSkillBuff.SetActive(leftPlayer.Skill.IsActive);
                RightDashBuff.SetActive(rightPlayer.IsDashActive);
                RightSkillBuff.SetActive(rightPlayer.Skill.IsActive);
            }
            else
            {
                ResetActionUI(LeftSkillUI, LeftDashUI, leftPlayer);
                ResetActionUI(RightSkillUI, RightDashUI, rightPlayer);

                // Reset timer UI
                if (Timer != null)
                    Timer.SetText(battle.BattleTime.ToString());
            }

            if (battle.CurrentState == BattleState.Battle_Countdown || battle.CurrentState == BattleState.Battle_Reset || battle.CurrentState == BattleState.PostBattle_ShowResult)
            {
                LeftDashBuff.SetActive(false);
                LeftSkillBuff.SetActive(false);
                RightDashBuff.SetActive(false);
                RightSkillBuff.SetActive(false);
            }
        }
        #endregion

        #region Battle changes
        private void OnBattleChanged(EventParameter param)
        {
            var battle = BattleManager.Instance.Battle;

            RoundSystem.SetText($"Best of {(int)battle.RoundSystem}");
            Round.SetText($"Round {battle.CurrentRound?.RoundNumber}");

            Round round = battle.CurrentRound;
            BattleState state = BattleManager.Instance.CurrentState;
            BattleStateUI.SetText(state.ToString().Replace("_", " "));

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
                    LeftInputType.value = BattleManager.Instance.LeftInputType.ToBattleInputType();
                    RightInputType.value = BattleManager.Instance.RightInputType.ToBattleInputType();
                    LeftSkill.value = (int)leftPlayer.Skill.Type;
                    RightSkill.value = (int)rightPlayer.Skill.Type;
                    LeftFinalScore.SetText("");
                    RightFinalScore.SetText("");

                    LeftCustomBtn.onClick.AddListener(() => CreateCostume(leftPlayer.Profile.ID));
                    RightCustomBtn.onClick.AddListener(() => CreateCostume(rightPlayer.Profile.ID));
                    break;
                case BattleState.Battle_Preparing:
                    battle.LeftPlayer.Events[SumoController.OnSkillAssigned].Subscribe(OnSkillAssigned);
                    battle.RightPlayer.Events[SumoController.OnSkillAssigned].Subscribe(OnSkillAssigned);

                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(true);
                    InitActionUI(LeftSkillUI, LeftDashUI, leftPlayer);
                    InitActionUI(RightSkillUI, RightDashUI, rightPlayer);
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
                    var leftSkill = leftPlayer.Skill;
                    var rightSkill = rightPlayer.Skill;

                    if (leftSkill.Type == SkillType.Boost)
                        LeftSkillBuff.GetComponentInChildren<TMP_Text>().text = $"SPD x {leftSkill.BoostMultiplier}";
                    else
                        LeftSkillBuff.GetComponentInChildren<TMP_Text>().text = "FREEZED";


                    if (rightSkill.Type == SkillType.Boost)
                        RightSkillBuff.GetComponentInChildren<TMP_Text>().text = $"SPD x {rightSkill.BoostMultiplier}";
                    else
                        RightSkillBuff.GetComponentInChildren<TMP_Text>().text = "FREEZED";

                    LeftDashBuff.GetComponentInChildren<TMP_Text>().text = $"SPD + {leftPlayer.DashSpeed}";
                    RightDashBuff.GetComponentInChildren<TMP_Text>().text = $"SPD + {rightPlayer.DashSpeed}";
                    BattleManager.Instance.Events[BattleManager.OnCountdownChanged].Subscribe(OnCountdownChanged);
                    break;
                case BattleState.Battle_Ongoing:
                    Countdown.SetText("");
                    BattleManager.Instance.Events[BattleManager.OnCountdownChanged].Unsubscribe(OnCountdownChanged);
                    break;
                case BattleState.Battle_End:
                    leftPlayer.Events[SumoController.OnSkillAssigned].Unsubscribe(OnSkillAssigned);
                    rightPlayer.Events[SumoController.OnSkillAssigned].Unsubscribe(OnSkillAssigned);
                    break;
                case BattleState.PostBattle_ShowResult:
                    SumoController winner = battle.GetBattleWinner().ToController(battle);
                    if (winner != null)
                    {
                        if (winner.Side == PlayerSide.Left)
                        {
                            LeftCrown.SetActive(true);
                            RightCrown.SetActive(false);
                        }
                        else
                        {
                            LeftCrown.SetActive(false);
                            RightCrown.SetActive(true);
                        }
                    }
                    else
                    {
                        LeftCrown.SetActive(false);
                        RightCrown.SetActive(false);
                    }

                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(true);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);

                    LeftFinalScore.SetText(battle.LeftWinCount.ToString());
                    RightFinalScore.SetText(battle.RightWinCount.ToString());
                    break;
            }

            PlayerHUD.ForEach((costume) =>
                   {
                       if (costume.Side == Placement.Left)
                           costume.AttachToUI(leftPlayer.Profile.CurrentCostume);
                       else if (costume.Side == Placement.Right)
                           costume.AttachToUI(rightPlayer.Profile.CurrentCostume);
                   });

            UpdateScore(battle);
        }

        private void OnCountdownChanged(EventParameter param)
        {
            Countdown.SetText(param.Float.ToString());
        }

        private void OnSkillAssigned(EventParameter param)
        {
            if (param.Side == PlayerSide.Left)
                LeftSkillUI.SetText(param.SkillType.ToString());
            else
                RightSkillUI.SetText(param.SkillType.ToString());
        }

        private void InitActionUI(CooldownUIGroupSet skill, CooldownUIGroupSet dash, SumoController controller)
        {

            InputType inputType = controller.InputProvider.InputType;

            skill.SetVisible(inputType);
            dash.SetVisible(inputType);
        }

        private void UpdateActionUI(CooldownUIGroupSet skill, CooldownUIGroupSet dash, SumoController player)
        {
            InputType inputType = player.InputProvider.InputType;

            skill.SetCooldown(player.Skill.CooldownNormalized, inputType);
            dash.SetCooldown(player.DashCooldownNormalized, inputType);
        }

        private void ResetActionUI(CooldownUIGroupSet skill, CooldownUIGroupSet dash, SumoController player)
        {
            InputType inputType = player.InputProvider.InputType;

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

        #region Battle player config
        public void CreateCostume(string id)
        {
            SFXManager.Instance.Play2D("ui_accept");
            GameManager.Instance.Battle_LoadCostumeScene(id);
        }

        public void SetDefaultSkill(PlayerSide side, int type)
        {
            SFXManager.Instance.Play2D("ui_accept");
            if (side == PlayerSide.Left)
                BattleManager.Instance.Battle.LeftPlayer.AssignSkill(type == 0 ? SkillType.Boost : SkillType.Stone);
            else
                BattleManager.Instance.Battle.RightPlayer.AssignSkill(type == 0 ? SkillType.Boost : SkillType.Stone);
        }

        public void SetInputMode(PlayerSide side, int type)
        {
            SFXManager.Instance.Play2D("ui_accept");
            InputType changedType;
            switch (type)
            {
                case 1:
                    changedType = InputType.LiveCommand;
                    break;
                case 2:
                    changedType = InputType.Script;
                    break;
                default:
                    changedType = InputType.UI;
                    break;
            }


            if (side == PlayerSide.Left)
            {
                EnableBotSelector(side, LeftScript, changedType == InputType.Script);
                BattleManager.Instance.LeftInputType = changedType;
            }
            else
            {
                EnableBotSelector(side, RightScript, changedType == InputType.Script);
                BattleManager.Instance.RightInputType = changedType;
            }
        }

        private void EnableBotSelector(PlayerSide side, TMP_Dropdown dropdown, bool isActive)
        {
            var botManager = BattleManager.Instance.BotManager;
            var botInstances = BotUtility.GetAllBotInstances();

            // Safety check: ensure we have bot instances
            if (botInstances == null || botInstances.Count == 0)
            {
                Debug.LogError("No Bot ScriptableObject instances found! Make sure Bot assets exist in the project.");
                return;
            }

            var botIDs = botInstances.ConvertAll(bot => bot.ID).ToArray();

            if (isActive)
            {
                dropdown.ClearOptions();
                dropdown.AddOptions(botIDs.ToList());
                dropdown.value = 0;
            }

            if (side == PlayerSide.Left)
            {
                LeftSkill.interactable = !isActive;
                botManager.LeftEnabled = isActive;
                if (isActive && botInstances.Count > 0)
                {
                    botManager.leftBotIndex = 0;
                    botManager.Left = botInstances[0];
                }
            }
            else
            {
                RightSkill.interactable = !isActive;
                botManager.RightEnabled = isActive;
                if (isActive && botInstances.Count > 0)
                {
                    botManager.rightBotIndex = 0;
                    botManager.Right = botInstances[0];
                }
            }
            dropdown.gameObject.SetActive(isActive);
        }

        private void OnBotSelect(PlayerSide side, int index)
        {
            SFXManager.Instance.Play2D("ui_accept");
            var botManager = BattleManager.Instance.BotManager;
            var botInstances = BotUtility.GetAllBotInstances();

            // Safety check
            if (botInstances == null || index >= botInstances.Count)
            {
                Debug.LogError($"Invalid bot index {index}. Available bots: {botInstances?.Count ?? 0}");
                return;
            }

            var bot = botInstances[index];
            if (side == PlayerSide.Left)
            {
                botManager.leftBotIndex = index;
                botManager.Assign(bot, PlayerSide.Left);
            }
            else
            {
                botManager.rightBotIndex = index;
                botManager.Assign(bot, PlayerSide.Right);
            }

        }
        #endregion

        #region Battle menu
        public void Pause()
        {
            SFXManager.Instance.Play2D("ui_accept");
            PausePanel.SetActive(true);
            Time.timeScale = 0;
        }

        public void Resume()
        {
            SFXManager.Instance.Play2D("ui_accept");
            PausePanel.SetActive(false);
            Time.timeScale = 1;
        }

        public void Restart()
        {
            SFXManager.Instance.Play2D("ui_accept");
            Time.timeScale = 1;
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }

		public void GoToMainMenu() {
			SFXManager.Instance.Play2D("ui_accept");
			Time.timeScale = 1;
			SceneManager.LoadScene("MainMenu");
		}

		public void ShowReplay()
        {
            SFXManager.Instance.Play2D("ui_accept");
            GameManager.Instance.Battle_ShowReplay();
        }

        public void ShowGuide()
        {
            SFXManager.Instance.Play2D("ui_accept");
            if (GuidePanel != null)
            {
                GuidePanel.SetActive(true);
                ShowGameplayGuide();
                ButtonGuide.SetActive(false);
                ButtonMainMenu.SetActive(false);
                ButtonResume.SetActive(false);
            }
        }

        public void HideGuide()
        {
            SFXManager.Instance.Play2D("ui_accept");
            if (GuidePanel != null) 
            {
                ButtonGuide.SetActive(true);
                ButtonMainMenu.SetActive(true);
                GuidePanel.SetActive(false);
                ButtonResume.SetActive(true);
            }
        }

        public void ShowGameplayGuide() => ShowGuide(GuideTab.Gameplay);
        public void ShowRulesGuide() => ShowGuide(GuideTab.Rules);
        public void ShowControlGuide() => ShowGuide(GuideTab.Control);

        private void ShowGuide(GuideTab tab)
        {
            SFXManager.Instance.Play2D("ui_accept");
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
