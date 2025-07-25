using SumoHelper;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
namespace SumoInput
{
    public class ButtonPointerHandler : MonoBehaviour, IPointerDownHandler, IPointerUpHandler
    {
        #region Action properties
        public EventRegistry Events = new();
        static public string OnHold = "ActionOnHold";
        static public string OnPress = "ActionOnPress";
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
                Events[OnPress]?.Invoke();
        }

        private void Update()
        {
            if (isHolding)
                if (GetComponent<Button>().interactable)
                    Events[OnHold]?.Invoke();
        }
        #endregion
    }
}