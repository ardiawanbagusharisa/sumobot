using TMPro;
using UnityEngine;

namespace BattleLoop
{
    public class BattleLoopUIManager : MonoBehaviour
    {
        public TMP_Text IndicatorBattle;
        public TMP_Text IndicatorBattleCountDownTimer;

        void Awake()
        {
            BattleLoopManager.Instance.OnPostStateChanged += OnStateChanged;
        }

        void OnDestroy()
        {
            BattleLoopManager.Instance.OnPostStateChanged -= OnStateChanged;
        }

        private void OnStateChanged(BattleState state)
        {
            IndicatorBattle.SetText(state.ToString());

            switch (state)
            {
                case BattleState.Countdown:
                    BattleLoopManager.Instance.OnCountdownChanged += OnCountdownChanged;
                    break;
                case BattleState.Battle:
                    IndicatorBattleCountDownTimer.SetText("");
                    BattleLoopManager.Instance.OnCountdownChanged -= OnCountdownChanged;
                    break;
            }
        }


        private void OnCountdownChanged(float timer)
        {
            IndicatorBattleCountDownTimer.SetText(timer.ToString());
        }
    }
}