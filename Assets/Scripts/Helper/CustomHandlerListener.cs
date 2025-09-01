
using SumoHelper;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class CustomHandlerListener : MonoBehaviour, IBeginDragHandler, IEndDragHandler, IScrollHandler, IPointerDownHandler, IPointerUpHandler
{
    public EventRegistry Events = new();
    public const string OnDrag = "OnDrag";
    public const string OnScrolling = "OnScroll";
    public const string OnHold = "OnHold";
    public const string OnPressDown = "OnPressDown";
    public const string OnPressUp = "OnPressUp";
    private bool isHolding = false;

    private Button attachedButton;

    void OnEnable()
    {
        attachedButton = GetComponent<Button>();
    }

    void Update()
    {
        // Handle holding logic
        if (isHolding)
            if (attachedButton != null && attachedButton.interactable)
                Events[OnHold]?.Invoke();
    }

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
        if (attachedButton != null)
        {
            if (attachedButton.interactable)
                Events[OnPressUp]?.Invoke();
        }
        else
            Events[OnPressUp]?.Invoke();
    }

    public void OnPointerDown(PointerEventData eventData)
    {
        isHolding = true;
        if (attachedButton != null)
        {
            if (attachedButton.interactable)
                Events[OnPressDown]?.Invoke();
        }
        else
            Events[OnPressDown]?.Invoke();

    }
}