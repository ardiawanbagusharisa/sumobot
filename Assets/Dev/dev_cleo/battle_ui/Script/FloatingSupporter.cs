using UnityEngine;

public class FloatingSupporter : MonoBehaviour
{
    private RectTransform rect;
    private RectTransform panelBounds;
    private Vector2 targetPos;
    private float spacing = 70f;

    public float speed = 30f;

    public void Init(RectTransform bounds, float minSpacing)
    {
        rect = GetComponent<RectTransform>();
        panelBounds = bounds;
        spacing = minSpacing;
        PickNewTarget();
    }

    void Update()
    {
        if (rect == null || panelBounds == null)
            return;

        rect.anchoredPosition = Vector2.MoveTowards(rect.anchoredPosition, targetPos, speed * Time.deltaTime);

        if (Vector2.Distance(rect.anchoredPosition, targetPos) < 1f)
        {
            PickNewTarget();
        }
    }

    private void PickNewTarget()
    {
        Vector2 size = panelBounds.rect.size;
        float x = Random.Range(-size.x / 2f + 30f, size.x / 2f - 30f);
        float y = Random.Range(-size.y / 2f + 30f, size.y / 2f - 30f);
        targetPos = new Vector2(x, y);
    }
}
