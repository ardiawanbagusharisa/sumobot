using UnityEngine;
using UnityEngine.EventSystems;

public class UIHelper
{
    public static void StretchButtonToParent(GameObject gameObject)
    {
        RectTransform rectTransform = gameObject.GetComponent<RectTransform>();
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(1, 1);
        rectTransform.offsetMin = Vector2.zero;
        rectTransform.offsetMax = Vector2.zero;
        rectTransform.anchoredPosition = Vector2.zero;
    }

}