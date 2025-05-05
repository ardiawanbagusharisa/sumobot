using System.Collections;
using UnityEngine;

public class PostBattleReveal : MonoBehaviour
{
    [Header("UI Elements to Reveal")]
    public GameObject[] uiElements;

    [Header("Settings")]
    public float delayBetweenElements = 0.5f;

    private bool isRevealing = false;

    private void Start()
    {
        foreach (GameObject ui in uiElements)
        {
            if (ui != null)
                ui.SetActive(false);
        }
    }

    public void StartReveal()
    {
        if (!isRevealing)
            StartCoroutine(RevealElements());
    }

    IEnumerator RevealElements()
    {
        isRevealing = true;

        foreach (GameObject ui in uiElements)
        {
            if (ui != null)
            {
                ui.SetActive(true);
                yield return new WaitForSeconds(delayBetweenElements);
            }
        }

        isRevealing = false;
    }
}
