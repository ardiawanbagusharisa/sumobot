using System.Collections.Generic;
using System.Linq;
using SumoCore;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.SceneManagement;


namespace SumoManager
{
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
        public TMP_Text BattleStateUI;
        public TMP_Text Countdown;
        public TMP_Text RoundSystem;
        public TMP_Text Round;
        public TMP_Text Timer;

        [Header("Battle UI - Left Player")]
        public TMP_Text LeftScore;
        public TMP_Text LeftFinalScore;
        public Image LeftDashCooldown;
        public Image LeftSkillCooldown;
        public TMP_Text LeftSkillName;

        [Header("Battle UI - Right Player")]
        public TMP_Text RightScore;
        public TMP_Text RightFinalScore;
        public Image RightDashCooldown;
        public Image RightSkillCooldown;
        public TMP_Text RightSkillName;
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
            BattleManager battleTime = BattleManager.Instance;
            if (battleTime.CurrentState == BattleState.Battle_Ongoing ||
            battleTime.CurrentState == BattleState.Battle_End ||
            battleTime.CurrentState == BattleState.Battle_Reset)
            {
                SumoController leftPlayer = battleTime.Battle.LeftPlayer;
                LeftSkillCooldown.GetComponent<Image>().fillAmount = leftPlayer.Skill.CooldownNormalized;
                LeftDashCooldown.GetComponent<Image>().fillAmount = leftPlayer.DashCooldownNormalized;

                SumoController rightPlayer = battleTime.Battle.RightPlayer;
                RightSkillCooldown.GetComponent<Image>().fillAmount = rightPlayer.Skill.CooldownNormalized;
                RightDashCooldown.GetComponent<Image>().fillAmount = rightPlayer.DashCooldownNormalized;

                if (rightPlayer != null)
                {
                    if (RightSkillCooldown != null)
                        RightSkillCooldown.fillAmount = rightPlayer.Skill.CooldownNormalized;
                    if (RightDashCooldown != null)
                        RightDashCooldown.fillAmount = rightPlayer.DashCooldownNormalized;
                }
            }
            else
            {
                if (RightSkillCooldown != null)
                    RightSkillCooldown.fillAmount = 0;
                if (RightDashCooldown != null)
                    RightDashCooldown.fillAmount = 0;
                if (LeftSkillCooldown != null)
                    LeftSkillCooldown.fillAmount = 0;
                if (LeftDashCooldown != null)
                    LeftDashCooldown.fillAmount = 0;

                // Reset timer UI
                if (Timer != null)
                    Timer.SetText(battleTime.BattleTime.ToString());
            }
        }
        #endregion

        #region Battle changes
        private void OnBattleChanged(object[] args)
        {
            var battle = (Battle)args[0];
            RoundSystem.SetText($"Best of {(int)battle.RoundSystem}");
            Round.SetText($"Round {battle.CurrentRound.RoundNumber}");

            Round round = battle.CurrentRound;
            BattleState state = BattleManager.Instance.CurrentState;
            BattleStateUI.SetText(state.ToString());

            switch (state)
            {
                case BattleState.PreBatle_Preparing:
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(true);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    LeftSkill.value = (int)BattleManager.Instance.Battle.LeftPlayer.Skill.Type;
                    RightSkill.value = (int)BattleManager.Instance.Battle.LeftPlayer.Skill.Type;
                    LeftFinalScore.SetText("");
                    RightFinalScore.SetText("");
                    break;
                case BattleState.Battle_Preparing:
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(true);
                    ClearScore();
                    Countdown.SetText("");
                    RoundSystem.SetText("");
                    Round.SetText("");
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    break;
                case BattleState.Battle_Countdown:
                    LeftSkillName.SetText(battle.LeftPlayer.Skill.Type.ToString());
                    RightSkillName.SetText(battle.RightPlayer.Skill.Type.ToString());
                    BattleManager.Instance.Actions[BattleManager.OnCountdownChanged].Subscribe(OnCountdownChanged);
                    break;
                case BattleState.Battle_Ongoing:
                    Countdown.SetText("");
                    BattleManager.Instance.Actions[BattleManager.OnCountdownChanged].Unsubscribe(OnCountdownChanged);
                    break;
                case BattleState.Battle_End:
                    break;
                case BattleState.PostBattle_ShowResult:
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(true);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    LeftFinalScore.SetText(battle.LeftWinCount.ToString());
                    RightFinalScore.SetText(battle.RightWinCount.ToString());
                    break;
            }
            UpdateScore(battle);
        }

        private void OnCountdownChanged(object[] args)
        {
            Countdown.SetText(((float)args[0]).ToString());
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

        #region Guide Panel
        // === Guide Panel Fields ===
        public GameObject guidePanel;
        public TMP_Text guideTitleText;
        public TMP_Text guideContentText;
        public Button gameplayTab;
        public Button rulesTab;
        public Button controlsTab;
        public ScrollRect guideScrollRect;
        public Color guideActiveTabColor = new Color(1f, 0.89f, 0.62f);
        public Color guideInactiveTabColor = new Color(0.88f, 0.88f, 0.88f);
        [TextArea(2, 6)] public string gameplayContent;
        [TextArea(2, 6)] public string rulesContent;
        [TextArea(2, 6)] public string controlsContent;

        // === Guide Panel Methods ===
        public void ShowGuidePanel()
        {
            if (guidePanel != null)
            {
                guidePanel.SetActive(true);
                ShowGuideTab("Gameplay");
            }
        }

        public void HideGuidePanel()
        {
            if (guidePanel != null)
                guidePanel.SetActive(false);
        }

        public void ShowGuideTab(string tab)
        {
            switch (tab)
            {
                case "Gameplay":
                    guideTitleText.text = "Gameplay";
                    guideContentText.text = gameplayContent;
                    SetGuideTabHighlight(gameplayTab, true);
                    SetGuideTabHighlight(rulesTab, false);
                    SetGuideTabHighlight(controlsTab, false);
                    break;
                case "Rules":
                    guideTitleText.text = "Rules";
                    guideContentText.text = rulesContent;
                    SetGuideTabHighlight(gameplayTab, false);
                    SetGuideTabHighlight(rulesTab, true);
                    SetGuideTabHighlight(controlsTab, false);
                    break;
                case "Controls":
                    guideTitleText.text = "Controls";
                    guideContentText.text = controlsContent;
                    SetGuideTabHighlight(gameplayTab, false);
                    SetGuideTabHighlight(rulesTab, false);
                    SetGuideTabHighlight(controlsTab, true);
                    break;
            }
            if (guideScrollRect != null)
                StartCoroutine(ResetGuideScroll());
        }

        private IEnumerator ResetGuideScroll()
        {
            yield return null;
            guideScrollRect.verticalNormalizedPosition = 1f;
        }

        private void SetGuideTabHighlight(Button tab, bool active)
        {
            if (tab == null) return;
            var colors = tab.colors;
            colors.normalColor = active ? guideActiveTabColor : guideInactiveTabColor;
            colors.selectedColor = active ? guideActiveTabColor : guideInactiveTabColor;
            tab.colors = colors;

            Image tabImg = tab.GetComponent<Image>();
            if (tabImg != null)
                tabImg.color = active ? guideActiveTabColor : guideInactiveTabColor;
        }
        #endregion

        #region Pause Panel
        public GameObject pausePanel;

        // Methods for pause panel
        public void ShowPause()
        {
            pausePanel.SetActive(true);
            Time.timeScale = 0;
        }

        public void OnResume()
        {
            pausePanel.SetActive(false);
            Time.timeScale = 1;
        }

        public void OnRestart()
        {
            Time.timeScale = 1;
            SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        }

        public void OnBack()
        {
            Time.timeScale = 1;
            SceneManager.LoadScene("campaignScene");
        }


        public void OnOut()
        {
            Time.timeScale = 1;
            SceneManager.LoadScene("MainMenu");
        }
        #endregion

        #region Initialization
        private void Start()
        {
            // Setup guide tab listeners
            if (gameplayTab != null) gameplayTab.onClick.AddListener(() => ShowGuideTab("Gameplay"));
            if (rulesTab != null) rulesTab.onClick.AddListener(() => ShowGuideTab("Rules"));
            if (controlsTab != null) controlsTab.onClick.AddListener(() => ShowGuideTab("Controls"));
            if (guidePanel != null) guidePanel.SetActive(false);
            if (pausePanel != null) pausePanel.SetActive(false);
        }
        #endregion
    }
}
