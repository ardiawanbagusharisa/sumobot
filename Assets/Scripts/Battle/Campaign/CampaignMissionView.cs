using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace SumoCampaign
{
    // All controls are authored in the campaign scenes; this component only binds them.
    public sealed class CampaignMissionView : MonoBehaviour
    {
        public GameObject Panel;
        public TMP_Text Title;
        public TMP_Text Message;
        public TMP_Text Progress;
        public GameObject CompletionSticker;
        public TMP_Text CompletionStickerLabel;
        public Button StartButton;
        public Button RetryButton;
        public Button MenuButton;
        public Button[] AnswerButtons;
        public TMP_Text[] AnswerLabels;

        private CampaignLevelController owner;

        public void Bind(CampaignLevelController controller)
        {
            owner = controller;
            StartButton.onClick.AddListener(owner.StartMission);
            RetryButton.onClick.AddListener(owner.RetryMission);
            MenuButton.onClick.AddListener(owner.ReturnToMenu);
            for (int i = 0; i < AnswerButtons.Length; i++)
            {
                int answer = i;
                AnswerButtons[i].onClick.AddListener(() => owner.AnswerQuestion(answer));
            }
        }

        public void Show(string title, string message, bool start, bool retry, bool quiz)
        {
            Panel.SetActive(true);
            Title.text = title;
            Message.text = message;
            StartButton.gameObject.SetActive(start);
            RetryButton.gameObject.SetActive(retry);
            for (int i = 0; i < AnswerButtons.Length; i++)
                AnswerButtons[i].gameObject.SetActive(quiz && i < owner.Level.Answers.Length);
            if (quiz)
                for (int i = 0; i < owner.Level.Answers.Length && i < AnswerLabels.Length; i++)
                    AnswerLabels[i].text = owner.Level.Answers[i];
        }

        public void Hide()
        {
            Panel.SetActive(false);
        }

        public void ShowCompletionSticker()
        {
            CompletionStickerLabel.text = "MISSION COMPLETE";
            CompletionSticker.SetActive(true);
        }

        public void HideCompletionSticker()
        {
            CompletionSticker.SetActive(false);
        }
    }
}
