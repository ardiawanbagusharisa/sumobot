using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class ButtonPointerHandler : MonoBehaviour, IPointerDownHandler, IPointerUpHandler
{
    public event Action OnHold;
    public event Action OnPress;
    private bool isHolding = false;

    public void OnPointerDown(PointerEventData eventData)
    {
        isHolding = true;
        Debug.Log("Button Down");
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        isHolding = false;
        OnPress?.Invoke();
        Debug.Log("Button Up");
    }

    private void Update()
    {
        if (isHolding)
        {
            OnHold?.Invoke();
        }
    }
}