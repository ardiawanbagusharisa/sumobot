using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class MyTabs : MonoBehaviour
{

    public GameObject tabbuttoncamp;
    public GameObject tabbuttonmult;
    
    public GameObject tabcontentcamp;
    public GameObject tabcontentmult;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void HideAllTabs()
    {
        tabcontentcamp.SetActive(false);
        tabcontentmult.SetActive(false);

        tabbuttoncamp.GetComponent<Button>().image.color = new Color32(212, 212, 212, 255);
        tabbuttonmult.GetComponent<Button>().image.color = new Color32(212, 212, 212, 255);
    }

    public void ShowTabComp()
    {
        HideAllTabs();
        tabcontentcamp.SetActive(true);
        tabbuttoncamp.GetComponent<Button>().image.color = new Color32(255, 255, 255, 255);
    }

    public void ShowTabMult()
    {
        HideAllTabs();
        tabcontentmult.SetActive(true);
        tabbuttonmult.GetComponent<Button>().image.color = new Color32(255, 255, 255, 255);
    }
}