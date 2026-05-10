
using TMPro;
using UnityEngine;

public class CostumeManager : MonoBehaviour
{
    public TMP_Text PlayerTxt;

    void Start()
    {
        PlayerTxt.text = $"{GameManager.Instance.GetProfileById().Name}'s Bot Preview ";
    }

    public void SaveAndBack()
    {
        SFXManager.Instance.Play2D("ui_accept");
        GameManager.Instance.BotCreator_SaveAndBack();
    }

    public void BackToBattle()
    {
		SFXManager.Instance.Play2D("ui_accept");
		UnityEngine.SceneManagement.SceneManager.LoadScene("Battle");
	}

    public void GoToModule() {
		SFXManager.Instance.Play2D("ui_accept");
		UnityEngine.SceneManagement.SceneManager.LoadScene("Dev/Bagus/BotModulesCreator");
	}
}