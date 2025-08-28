
using SumoHelper;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class CustomHandlerListener : MonoBehaviour, IBeginDragHandler, IEndDragHandler, IScrollHandler, IPointerDownHandler, IPointerUpHandler
{
    public EventRegistry Events = new();
    public const string OnDrag = "OnDrag";
    public const string OnScrolling = "OnScroll";
    public const string OnHold = "ActionOnHold";
    public const string OnPress = "ActionOnPress";
    private bool isHolding = false;

    private Button attachedButton;
    public void OnBeginDrag(PointerEventData eventData)
    {
        Events[OnDrag]?.Invoke(new(boolParam: true));
    }

    public void OnEndDrag(PointerEventData eventData)
    {
        Events[OnDrag]?.Invoke(new(boolParam: false));
    }

    public void OnScroll(PointerEventData eventData)
    {
        Events[OnScrolling]?.Invoke();
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        isHolding = false;
        if (attachedButton != null && attachedButton.interactable)
            Events[OnPress]?.Invoke();
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        isHolding = true;
    }

    void Update()
    {
        if (isHolding)
            if (attachedButton != null && attachedButton.interactable)
                Events[OnHold]?.Invoke();
    }

    void OnEnable()
    {
        attachedButton = GetComponent<Button>();
    }
}