using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class ButtonPointerHandler : MonoBehaviour, IPointerDownHandler, IPointerUpHandler
{
    #region Action properties
    public event Action OnHold;
    public event Action OnPress;
    private bool isHolding = false;
    #endregion

    #region Unity methods
    public void OnPointerDown(PointerEventData eventData)
    {
        isHolding = true;
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        isHolding = false;
        if (GetComponent<Button>().interactable)
        {
            OnPress?.Invoke();
        }
    }

    private void Update()
    {
        if (isHolding)
        {
            if (GetComponent<Button>().interactable)
            {
                OnHold?.Invoke();
            }
        }
    }
    #endregion
}