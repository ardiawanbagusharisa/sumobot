using System.Collections;
using UnityEngine;

public class DamageSmoke : MonoBehaviour
{
    public int Hit = 0;
    public ProceduralSmoke2D smokeController;

    private bool isProcessing = false;

    private void OnCollisionEnter2D(Collision2D collision)
    {
        // Optional: add filtering (e.g., only bullets or player)
        if (!isProcessing)
        {
            Hit++;
            StartCoroutine(ApplyDamageStateAfterDelay(0.5f));
        }
    }

    private IEnumerator ApplyDamageStateAfterDelay(float delay)
    {
        isProcessing = true;
        yield return new WaitForSeconds(delay);

        if (smokeController == null)
        {
            isProcessing = false;
            yield break;
        }

        if (Hit <= 0)
        {
            smokeController.gameObject.SetActive(false);
        }
        else
        {
            smokeController.gameObject.SetActive(true);

            if (Hit == 1)
                smokeController.currentDamageState = ProceduralSmoke2D.DamageState.Normal;
            else if (Hit == 2)
                smokeController.currentDamageState = ProceduralSmoke2D.DamageState.HeavySmoke;
            else
                smokeController.currentDamageState = ProceduralSmoke2D.DamageState.Fire;
        }

        isProcessing = false;
    }
}