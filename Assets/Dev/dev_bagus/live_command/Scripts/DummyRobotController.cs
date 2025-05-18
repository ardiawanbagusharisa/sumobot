using UnityEngine;

public class RobotController : MonoBehaviour
{
    public float moveSpeed = 5f;
    public float rotationSpeed = 90f;
    public float dashSpeed = 15f;

    private int battles = 10, wins = 6, losses = 4;
    private int winByDash = 2, winBySkill = 1;
    private float gameTimer = 120f;
    private int rounds = 3;

    public void Accelerate(float power, float duration)
    {
        StartCoroutine(AccelerateCoroutine(power, duration));
    }

    private System.Collections.IEnumerator AccelerateCoroutine(float power, float duration)
    {
        float elapsed = 0f;
        while (elapsed < duration)
        {
            transform.Translate(power * Time.deltaTime * Vector3.forward);
            elapsed += Time.deltaTime;
            yield return null;
        }
    }

    public void TurnLeft(float angle)
    {
        transform.Rotate(0, 0, -angle);
    }

    public void TurnRight(float angle)
    {
        transform.Rotate(0, 0, angle);
    }

    public void Dash()
    {
        transform.Translate(Vector3.forward * dashSpeed * Time.deltaTime);
    }

    public void SpecialSkill()
    {
        // Implement a unique skill here
        Debug.Log("Special skill activated!");
    }

    public string GetStatus()
    {
        return
            $"Robot Stats:\n- Move Speed: {moveSpeed}\n- Rotation Speed: {rotationSpeed}\n- Dash Speed: {dashSpeed}\n" +
            $"Player Stats:\n- Battles: {battles}\n- Wins: {wins}\n- Losses: {losses}\n- Win by Dash: {winByDash}\n- Win by Skill: {winBySkill}\n" +
            $"Game Stats:\n- Timer: {gameTimer}\n- Rounds: {rounds}";
    }
}
