using System.Threading;
using TMPro;
using UnityEngine;

namespace BattleLoop
{
    public class BattleUIManager : MonoBehaviour
    {
        public TMP_Text IndicatorBattle;
        public TMP_Text IndicatorBattleCountDownTimer;
        public TMP_Text LeftScore;
        public TMP_Text BattleTime;
        public TMP_Text RightScore;

        void Awake()
        {
            BattleManager.Instance.OnBattleInfoChanged += OnBattleInfoChanged;
            BattleManager.Instance.OnPostStateChanged += OnStateChanged;
        }

        void OnDestroy()
        {
            BattleManager.Instance.OnPostStateChanged -= OnStateChanged;
            BattleManager.Instance.OnBattleInfoChanged -= OnBattleInfoChanged;
        }

        private void OnStateChanged(BattleState state)
        {
            IndicatorBattle.SetText(state.ToString());

            switch (state)
            {
                case BattleState.Countdown:
                    BattleManager.Instance.OnCountdownChanged += OnCountdownChanged;
                    break;
                case BattleState.Battle:
                    IndicatorBattleCountDownTimer.SetText("");
                    BattleManager.Instance.OnCountdownChanged -= OnCountdownChanged;
                    break;
            }
        }


        private void OnBattleInfoChanged(BattleInfo info)
        {
            if (info.LeftPlayer != null)
                LeftScore.SetText(info.LeftPlayer.Score.ToString());
            if (info.RightPlayer != null)
                RightScore.SetText(info.RightPlayer.Score.ToString());

            BattleTime.SetText(info.Time.ToString());
        }


        private void OnCountdownChanged(float timer)
        {
            IndicatorBattleCountDownTimer.SetText(timer.ToString());
        }
    }
}