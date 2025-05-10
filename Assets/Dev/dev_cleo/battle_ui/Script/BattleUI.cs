using UnityEngine;

public class BattleUI : MonoBehaviour
{
    [SerializeField] private ScoreDot[] player1Dots;
    [SerializeField] private ScoreDot[] player2Dots;

    public void UpdateScoreUI(int p1Score, int p2Score)
    {
        UpdateDots(player1Dots, p1Score);
        UpdateDots(player2Dots, p2Score);
    }

    private void UpdateDots(ScoreDot[] dots, int score)
    {
        for (int i = 0; i < dots.Length; i++)
        {
            dots[i].SetActive(i < score);
        }
    }
}
