using SumoHelper;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
namespace SumoInput
{
    public class ButtonPointerHandler : MonoBehaviour, IPointerDownHandler, IPointerUpHandler
    {
        #region Action properties
        public EventRegistry Actions = new();
        static public string ActionOnHold = "ActionOnHold";
        static public string ActionOnPress = "ActionOnPress";
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
                Actions[ActionOnPress]?.Invoke();
        }

        private void Update()
        {
            if (isHolding)
                if (GetComponent<Button>().interactable)
                    Actions[ActionOnHold]?.Invoke();
        }
        #endregion
    }
}