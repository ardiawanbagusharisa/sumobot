using TMPro;
using UnityEngine;
using System.Collections.Generic;

public class MarketDropdownHandler : MonoBehaviour
{
    [System.Serializable]
    public class MarketDropdown
    {
        public TMP_Dropdown dropdown;
        public string dropdownName; // Misal: Aesthetic, Gameplay, Crafting
    }

    public MarketDropdown[] allDropdowns;

    [System.Serializable]
    public class SubcategoryPanel
    {
        public string subcategoryName;  // Harus cocok dengan opsi dropdown
        public GameObject panelObject;  // Panel yang akan ditampilkan
    }

    public List<SubcategoryPanel> subcategoryPanels;

    void Start()
    {
        foreach (var d in allDropdowns)
        {
            d.dropdown.onValueChanged.AddListener((index) => OnDropdownChanged(d, index));
        }
    }

    void OnDropdownChanged(MarketDropdown marketDropdown, int index)
    {
        if (index == 0)
        {
            marketDropdown.dropdown.value = 0;
            return;
        }

        string selectedOption = marketDropdown.dropdown.options[index].text;
        Debug.Log($"{marketDropdown.dropdownName} → {selectedOption}");

        // Ganti panel konten
        ShowPanelForSubcategory(selectedOption);

        // Reset dropdown ke default label
        marketDropdown.dropdown.value = 0;
    }

    void ShowPanelForSubcategory(string subcategory)
    {
        foreach (var entry in subcategoryPanels)
        {
            entry.panelObject.SetActive(entry.subcategoryName == subcategory);
        }
    }
}
