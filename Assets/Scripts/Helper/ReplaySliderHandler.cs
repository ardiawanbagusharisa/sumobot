using UnityEngine;
using UnityEngine.EventSystems;

public class ReplaySliderHandler : MonoBehaviour, IPointerDownHandler, IPointerUpHandler
{
    public void OnPointerDown(PointerEventData eventData)
    {
        ReplayManager.Instance.OnTimeSliderPointerDown();
    }

    public void OnPointerUp(PointerEventData eventData)
    {
        ReplayManager.Instance.OnTimeSliderPointerUp();
    }
}
