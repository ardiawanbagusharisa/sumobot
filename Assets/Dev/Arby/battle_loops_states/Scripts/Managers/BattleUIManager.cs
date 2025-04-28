using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace BattleLoop
{
    public class BattleUIManager : MonoBehaviour
    {
        public TMP_Text IndicatorBattle;
        public TMP_Text IndicatorBattleCountDownTimer;
        public TMP_Text BattleTime;
        public GameObject LeftScore;
        public GameObject RightScore;

        public List<GameObject> BattleStatePanel = new List<GameObject>();



        private List<Image> leftScoreDots;
        private List<Image> rightScoreDots;

        void Awake()
        {
            BattleManager.Instance.OnBattleInfoChanged += OnBattleInfoChanged;
            BattleManager.Instance.OnRoundInfoChanged += OnBattleInfosChanged;

            leftScoreDots = LeftScore.GetComponentsInChildren<Image>()
            .Where(img => img.gameObject != LeftScore).ToList();

            rightScoreDots = RightScore.GetComponentsInChildren<Image>().Where(img => img.gameObject != RightScore).ToList();
        }

        void OnDestroy()
        {
            BattleManager.Instance.OnBattleInfoChanged -= OnBattleInfoChanged;
            BattleManager.Instance.OnRoundInfoChanged -= OnBattleInfosChanged;
        }

        private void OnBattleInfoChanged(BattleInfo info)
        {
            BattleTime.SetText(info.Time.ToString());

            IndicatorBattle.SetText(info.battleState.ToString());

            switch (info.battleState)
            {
                case BattleState.PreBatle_Preparing:
                    IndicatorBattleCountDownTimer.SetText("");
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(true);

                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    break;

                case BattleState.Battle_Preparing:
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(true);

                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Post")).SetActive(false);
                    ClearScore();
                    break;
                case BattleState.Battle_Countdown:
                    BattleManager.Instance.OnCountdownChanged += OnCountdownChanged;
                    break;
                case BattleState.Battle_Ongoing:
                    IndicatorBattleCountDownTimer.SetText("");
                    BattleManager.Instance.OnCountdownChanged -= OnCountdownChanged;
                    break;
                case BattleState.Battle_End:
                    UpdateScore(info);
                    break;

                case BattleState.PostBattle_ShowResult:
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Post")).SetActive(true);

                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Ongoing")).SetActive(false);
                    BattleStatePanel.Find((o) => o.CompareTag("BattleState/Pre")).SetActive(false);
                    break;
            }
        }


        private void OnCountdownChanged(float timer)
        {
            IndicatorBattleCountDownTimer.SetText(timer.ToString());
        }

        private void OnBattleInfosChanged(Dictionary<int, BattleInfo> infos)
        {
        }

        private void UpdateScore(BattleInfo battleInfo)
        {
            if (battleInfo.WinnerEachRound.Count() == 0)
            {
                leftScoreDots.ForEach(x => x.color = Color.white);
                rightScoreDots.ForEach(x => x.color = Color.white);
                return;
            }

            foreach (var winner in battleInfo.WinnerEachRound)
            {
                // if (currRound == 0) { continue; }

                switch (winner.Value)
                {
                    case BattleWinner.Draw:
                        leftScoreDots[winner.Key - 1].color = Color.grey;
                        rightScoreDots[winner.Key - 1].color = Color.grey;
                        break;
                    case BattleWinner.Ongoing:
                        leftScoreDots[winner.Key - 1].color = Color.white;
                        rightScoreDots[winner.Key - 1].color = Color.white;
                        break;
                    case BattleWinner.Left:
                        leftScoreDots[winner.Key - 1].color = Color.green;
                        rightScoreDots[winner.Key - 1].color = Color.black;
                        break;
                    case BattleWinner.Right:
                        leftScoreDots[winner.Key - 1].color = Color.black;
                        rightScoreDots[winner.Key - 1].color = Color.green;
                        break;
                }
            }
        }
        private void ClearScore()
        {

            leftScoreDots.ForEach(x => x.color = Color.white);
            rightScoreDots.ForEach(x => x.color = Color.white);
        }
    }
}
