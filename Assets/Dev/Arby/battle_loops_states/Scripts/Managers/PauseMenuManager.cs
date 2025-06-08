using UnityEngine;
using UnityEngine.SceneManagement;

public class PausePanelManager : MonoBehaviour
{
    public GameObject PausePanel;

    public void ShowPause()
    {
        PausePanel.SetActive(true);
        Time.timeScale = 0;
    }

    public void OnResume()
    {
        PausePanel.SetActive(false);
        Time.timeScale = 1;
    }

    public void OnRestart()
    {
        Time.timeScale = 1;
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    public void OnBack()
    {
        Time.timeScale = 1;
        SceneManager.LoadScene("campaignScene");
    }


    public void OnOut()
    {
        Time.timeScale = 1;
        SceneManager.LoadScene("MainMenu");
    }
}
