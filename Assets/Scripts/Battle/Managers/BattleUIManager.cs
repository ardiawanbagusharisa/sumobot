using CoreSumo;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

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
        public TMP_Text BattleState;
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

        void OnEnable()
        {
            BattleManager.Instance.OnBattleChanged += OnBattleChanged;
        }

        void OnDisable()
        {
            BattleManager.Instance.OnBattleChanged -= OnBattleChanged;
        }

        void FixedUpdate()
        {
            if (BattleManager.Instance.CurrentState == global::BattleState.Battle_Ongoing ||
            BattleManager.Instance.CurrentState == global::BattleState.Battle_End ||
            BattleManager.Instance.CurrentState == global::BattleState.Battle_Reset)
            {
                SumoController leftPlayer = BattleManager.Instance.Battle.LeftPlayer;
                LeftSkillCooldown.GetComponent<Image>().fillAmount = leftPlayer.Skill.CooldownNormalized;
                LeftDashCooldown.GetComponent<Image>().fillAmount = leftPlayer.DashCooldownNormalized;

                SumoController rightPlayer = BattleManager.Instance.Battle.RightPlayer;
                RightSkillCooldown.GetComponent<Image>().fillAmount = rightPlayer.Skill.CooldownNormalized;
                RightDashCooldown.GetComponent<Image>().fillAmount = rightPlayer.DashCooldownNormalized;

                Timer.SetText(Mathf.CeilToInt(BattleManager.Instance.TimeLeft).ToString());
            }
            else
            {
                RightSkillCooldown.GetComponent<Image>().fillAmount = 0;
                RightDashCooldown.GetComponent<Image>().fillAmount = 0;
                LeftSkillCooldown.GetComponent<Image>().fillAmount = 0;
                LeftDashCooldown.GetComponent<Image>().fillAmount = 0;

                Timer.SetText(BattleManager.Instance.BattleTime.ToString());
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
            BattleState.SetText(state.ToString());

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
    }
}

