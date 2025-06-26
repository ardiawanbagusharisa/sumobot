using CoreSumo;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.SceneManagement;

namespace BattleLoop
{
    public class BattleUIManager : MonoBehaviour
    {
        public static BattleUIManager Instance { get; private set; }

        #region UI Elements properties
        [Header("Main Panels")]
        public List<GameObject> BattlePanels = new List<GameObject>();

        [Header("Pre-battle UI")]
        public TMP_Dropdown LeftSkill;
        public TMP_Dropdown RightSkill;

        [Header("Battle UI")]
        public TMP_Text BattleStateText;

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
            if (BattleManager.Instance != null)
                BattleManager.Instance.OnBattleChanged += OnBattleChanged;
        }

        private void OnDisable()
        {
            if (BattleManager.Instance != null)
                BattleManager.Instance.OnBattleChanged -= OnBattleChanged;
        }

        private void FixedUpdate()
        {
            if (BattleManager.Instance == null) return;

            var state = BattleManager.Instance.CurrentState;
            bool isBattleState = state == BattleState.Battle_Ongoing ||
                                 state == BattleState.Battle_End ||
                                 state == BattleState.Battle_Reset;

            int timeLeft = Mathf.CeilToInt(BattleManager.Instance.TimeLeft);
            float battleTime = BattleManager.Instance.BattleTime;

            if (isBattleState)
            {
                if (Timer != null)
                    Timer.SetText(timeLeft.ToString());

                var leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
                if (leftPlayer != null)
                {
                    if (LeftSkillCooldown != null)
                        LeftSkillCooldown.fillAmount = leftPlayer.Skill.CooldownAmountNormalized;
                    if (LeftDashCooldown != null)
                        LeftDashCooldown.fillAmount = leftPlayer.DashCooldownNormalized;
                }

                var rightPlayer = BattleManager.Instance.Battle.RightPlayer;
                if (rightPlayer != null)
                {
                    if (RightSkillCooldown != null)
                        RightSkillCooldown.fillAmount = rightPlayer.Skill.CooldownAmountNormalized;
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
                    Timer.SetText(battleTime.ToString());
            }
        }
        #endregion

        #region Battle changes
        private void OnBattleChanged(Battle battle)
        {
            RoundSystem.SetText($"Best of {(int)battle.RoundSystem}");
            Round.SetText($"Round {battle.CurrentRound.RoundNumber}");

            Round round = battle.CurrentRound;
            BattleState state = BattleManager.Instance.CurrentState;
            BattleStateText.SetText(state.ToString());

            switch (state)
            {
                case global::BattleState.PreBatle_Preparing:
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(true);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    LeftSkill.value = (int)BattleManager.Instance.Battle.LeftPlayer.Skill.Type;
                    RightSkill.value = (int)BattleManager.Instance.Battle.LeftPlayer.Skill.Type;
                    LeftFinalScore.SetText("");
                    RightFinalScore.SetText("");
                    break;
                case global::BattleState.Battle_Preparing:
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(true);
                    ClearScore();
                    Countdown.SetText("");
                    RoundSystem.SetText("");
                    Round.SetText("");
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    break;
                case global::BattleState.Battle_Countdown:
                    LeftSkillName.SetText(BattleManager.Instance.Battle.LeftPlayer.Skill.Type.ToString());
                    RightSkillName.SetText(BattleManager.Instance.Battle.RightPlayer.Skill.Type.ToString());
                    BattleManager.Instance.OnCountdownChanged += OnCountdownChanged;
                    break;
                case global::BattleState.Battle_Ongoing:
                    Countdown.SetText("");
                    BattleManager.Instance.OnCountdownChanged -= OnCountdownChanged;
                    break;
                case global::BattleState.Battle_End:
                    InputManager.Instance.ResetCooldownButton();
                    break;
                case global::BattleState.PostBattle_ShowResult:
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Post")).SetActive(true);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattlePanels.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    LeftFinalScore.SetText(battle.LeftWinCount.ToString());
                    RightFinalScore.SetText(battle.RightWinCount.ToString());
                    break;
            }
            UpdateScore(battle);
        }

        private void OnCountdownChanged(float timer)
        {
            Countdown.SetText(timer.ToString());
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

