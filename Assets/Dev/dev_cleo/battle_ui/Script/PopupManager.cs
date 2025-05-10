using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class FlexiblePopupManager : MonoBehaviour
{
    [System.Serializable]
    public class ButtonPopupPair
    {
        public Button button;
        public GameObject popup;
    }

    public List<ButtonPopupPair> buttonPopupPairs;

    private GameObject currentPopup;

    void Start()
    {
        foreach (var pair in buttonPopupPairs)
        {
            pair.popup.SetActive(false); // Sembunyikan semua popup dulu
            pair.button.onClick.AddListener(() => TogglePopup(pair.popup));
        }
    }

    void TogglePopup(GameObject targetPopup)
    {
        if (currentPopup == targetPopup)
        {
            targetPopup.SetActive(false);
            currentPopup = null;
        }
        else
        {
            if (currentPopup != null) currentPopup.SetActive(false);
            targetPopup.SetActive(true);
            currentPopup = targetPopup;
        }
    }
}
