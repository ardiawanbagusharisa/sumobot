using TMPro;
using UnityEngine;

namespace BattleLoop
{
    public class BattleUIManager : MonoBehaviour
    {
        public TMP_Text IndicatorBattle;
        public TMP_Text IndicatorBattleCountDownTimer;

        void Awake()
        {
            BattleManager.Instance.OnPostStateChanged += OnStateChanged;
        }

        void OnDestroy()
        {
            BattleManager.Instance.OnPostStateChanged -= OnStateChanged;
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


        private void OnCountdownChanged(float timer)
        {
            IndicatorBattleCountDownTimer.SetText(timer.ToString());
        }
    }
}