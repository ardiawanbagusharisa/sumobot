using System;
using System.Collections.Generic;
using System.Linq;
using CoreSumoRobot;
using TMPro;
using Unity.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace BattleLoop
{

    public class BattleUIManager : MonoBehaviour
    {
        public static BattleUIManager Instance { get; private set; }

        public List<GameObject> BattleStatePanel = new List<GameObject>();

        // Pre-battle
        public TMP_Dropdown PreLeftDefaultSkill;
        public TMP_Dropdown PreRightDefaultSkill;

        // Battle
        public TMP_Text IndicatorBattle;
        public TMP_Text IndicatorBattleCountDownTimer;
        public TMP_Text StageBestOf;
        public TMP_Text StageRoundNumber;
        public TMP_Text StageBattleTime;
        public GameObject LeftScore;
        public GameObject RightScore;
        private List<Image> leftScoreDots = new List<Image>();
        private List<Image> rightScoreDots = new List<Image>();

        private void Awake()
        {
            if (Instance != null)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;

        }

        void Start()
        {
            BattleManager.Instance.OnBattleChanged += OnBattleChanged;


            leftScoreDots = LeftScore.GetComponentsInChildren<Image>().Where(img => img.gameObject != LeftScore).ToList();
            // Reverse the left player scores because we want the indicators start from right
            leftScoreDots.Reverse();

            rightScoreDots = RightScore.GetComponentsInChildren<Image>().Where(img => img.gameObject != RightScore).ToList();
        }

        void OnDestroy()
        {
            BattleManager.Instance.OnBattleChanged -= OnBattleChanged;
        }

        private void OnBattleChanged(Battle battle)
        {
            StageBestOf.SetText($"Best of {(int)battle.RoundSystem}");
            StageRoundNumber.SetText($"Round {battle.CurrentRound.RoundNumber}");

            Round round = battle.CurrentRound;
            StageBattleTime.SetText(round.TimeLeft.ToString());
            IndicatorBattle.SetText(round.BattleState.ToString());

            switch (round.BattleState)
            {
                case BattleState.PreBatle_Preparing:
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(true);

                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);

                    PreLeftDefaultSkill.value = (int)BattleManager.Instance.Battle.LeftPlayer.Skill.Type;
                    PreRightDefaultSkill.value = (int)BattleManager.Instance.Battle.LeftPlayer.Skill.Type;
                    break;

                case BattleState.Battle_Preparing:
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(true);
                    ClearScore();
                    IndicatorBattleCountDownTimer.SetText("");
                    StageBestOf.SetText("");
                    StageRoundNumber.SetText("");

                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);

                    // ClearScore();
                    break;
                case BattleState.Battle_Countdown:
                    BattleManager.Instance.OnCountdownChanged += OnCountdownChanged;
                    break;
                case BattleState.Battle_Ongoing:
                    IndicatorBattleCountDownTimer.SetText("");
                    BattleManager.Instance.OnCountdownChanged -= OnCountdownChanged;
                    break;
                case BattleState.Battle_End:
                    break;

                case BattleState.PostBattle_ShowResult:
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Post")).SetActive(true);

                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    break;
            }
            UpdateScore(battle);
        }

        private void OnCountdownChanged(float timer)
        {
            IndicatorBattleCountDownTimer.SetText(timer.ToString());
        }

        private void UpdateScore(Battle battleInfo)
        {
            if (battleInfo.Winners.Count() == 0)
            {
                ClearScore();
                return;
            }

            for (int i = 1; i < leftScoreDots.Count; i++)
            {
                if (i <= battleInfo.LeftWinCount)
                {
                    if (!leftScoreDots[i - 1].IsDestroyed())
                        leftScoreDots[i - 1].color = Color.green;
                }
                if (i <= battleInfo.RightWinCount)
                {
                    if (!leftScoreDots[i - 1].IsDestroyed())
                        rightScoreDots[i - 1].color = Color.green;
                }
            }
        }

        private void ClearScore()
        {
            for (int i = 0; i < leftScoreDots.Count - 1; i++)
            {
                leftScoreDots[i].color = Color.white;
                rightScoreDots[i].color = Color.white;
            }
        }
    }
}

class FilledScore
{
    public Image ScoreImage;
    public bool IsFilled;

    public FilledScore(Image scoreImage, bool isFilled)
    {
        ScoreImage = scoreImage;
        IsFilled = isFilled;
    }
}

