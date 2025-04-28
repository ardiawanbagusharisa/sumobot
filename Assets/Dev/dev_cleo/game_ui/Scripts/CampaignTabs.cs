using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class CampaignTabs : MonoBehaviour
{
    [Header("Buttons")]
    public Button btnTutorial;
    public Button btnC1;
    public Button btnC2;
    public Button btnC3;
    public Button btnC4;

    [Header("Contents")]
    public GameObject contentTutorial;
    public GameObject contentC1;
    public GameObject contentC2;
    public GameObject contentC3;
    public GameObject contentC4;

    [Header("Button Colors")]
    public Color activeColor = Color.white;
    public Color inactiveColor = new Color32(212, 212, 212, 255);

    void Start()
    {
        // Set listeners
        btnTutorial.onClick.AddListener(ShowTutorial);
        btnC1.onClick.AddListener(ShowC1);
        btnC2.onClick.AddListener(ShowC2);
        btnC3.onClick.AddListener(ShowC3);
        btnC4.onClick.AddListener(ShowC4);

        // Default tab
        ShowTutorial();
    }

    void HideAll()
    {
        contentTutorial.SetActive(false);
        contentC1.SetActive(false);
        contentC2.SetActive(false);
        contentC3.SetActive(false);
        contentC4.SetActive(false);

        btnTutorial.image.color = inactiveColor;
        btnC1.image.color = inactiveColor;
        btnC2.image.color = inactiveColor;
        btnC3.image.color = inactiveColor;
        btnC4.image.color = inactiveColor;
    }

    void ShowTutorial()
    {
        HideAll();
        contentTutorial.SetActive(true);
        btnTutorial.image.color = activeColor;
    }

    void ShowC1()
    {
        HideAll();
        contentC1.SetActive(true);
        btnC1.image.color = activeColor;
    }

    void ShowC2()
    {
        HideAll();
        contentC2.SetActive(true);
        btnC2.image.color = activeColor;
    }

    void ShowC3()
    {
        HideAll();
        contentC3.SetActive(true);
        btnC3.image.color = activeColor;
    }

    void ShowC4()
    {
        HideAll();
        contentC4.SetActive(true);
        btnC4.image.color = activeColor;
    }
}
