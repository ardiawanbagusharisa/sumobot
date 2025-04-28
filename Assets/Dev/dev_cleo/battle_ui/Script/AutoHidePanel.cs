using UnityEngine;

[RequireComponent(typeof(CanvasGroup))]
public class AutoHidePanel : MonoBehaviour
{
    public float showDuration = 3f;
    public float fadeDuration = 1f;
    public GameObject backgroundOverlay;

    private CanvasGroup canvasGroup;
    private bool isFading = false;

    private void Awake()
    {
        canvasGroup = GetComponent<CanvasGroup>();
        canvasGroup.alpha = 0f;
    }

    private void OnEnable()
    {
        canvasGroup.alpha = 1f;
        canvasGroup.interactable = true;
        canvasGroup.blocksRaycasts = true;
        isFading = false;

        if (backgroundOverlay != null)
            backgroundOverlay.SetActive(true);

        CancelInvoke();
        Invoke(nameof(StartFadeOut), showDuration);
    }

    void StartFadeOut()
    {
        if (!isFading)
            StartCoroutine(FadeOut());
    }

    System.Collections.IEnumerator FadeOut()
    {
        isFading = true;

        float startAlpha = canvasGroup.alpha;
        float time = 0f;

        while (time < fadeDuration)
        {
            time += Time.deltaTime;
            canvasGroup.alpha = Mathf.Lerp(startAlpha, 0f, time / fadeDuration);
            yield return null;
        }

        canvasGroup.alpha = 0f;
        canvasGroup.interactable = false;
        canvasGroup.blocksRaycasts = false;

        if (backgroundOverlay != null)
            backgroundOverlay.SetActive(false);

        gameObject.SetActive(false);
    }
}
