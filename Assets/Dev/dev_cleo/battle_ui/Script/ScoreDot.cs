using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class ScoreDot : MonoBehaviour
{
    [SerializeField] private Image dotImage;
    [Range(0f, 1f)]
    public float inactiveOpacity = 0.3f;
    public float fadeDuration = 0.25f;

    private void Start()
    {
        // Set opacity awal ke inactive
        SetAlpha(inactiveOpacity);
    }

    public void SetActive(bool isActive)
    {
        StopAllCoroutines();
        StartCoroutine(FadeTo(isActive ? 1f : inactiveOpacity));
    }

    private IEnumerator FadeTo(float targetAlpha)
    {
        float startAlpha = dotImage.color.a;
        float t = 0f;
        while (t < fadeDuration)
        {
            t += Time.deltaTime;
            float blend = Mathf.Clamp01(t / fadeDuration);
            float newAlpha = Mathf.Lerp(startAlpha, targetAlpha, blend);

            Color color = dotImage.color;
            color.a = newAlpha;
            dotImage.color = color;

            yield return null;
        }

        // Set ke nilai akhir biar akurat
        SetAlpha(targetAlpha);
    }

    private void SetAlpha(float alpha)
    {
        if (dotImage != null)
        {
            Color color = dotImage.color;
            color.a = alpha;
            dotImage.color = color;
        }
    }
}
