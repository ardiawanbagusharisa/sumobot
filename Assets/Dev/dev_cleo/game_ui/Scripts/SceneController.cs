using UnityEngine;
using UnityEngine.SceneManagement;

#if UNITY_STANDALONE_WIN
using System.Runtime.InteropServices;
#endif

public class SceneController : MonoBehaviour
{
    public SceneTransitionManager transitionManager;
    private void TransitionTo(string sceneName)
{
    if (transitionManager != null)
        transitionManager.StartSceneTransition(sceneName);
    else
        SceneManager.LoadScene(sceneName); // fallback kalau belum diset
}

    public void LoadMenu()
    {
        TransitionTo("menuScene");
    }

    public void LoadHome()
    {
        TransitionTo("homeScene");
    }

    public void LoadMarket()
    {
        TransitionTo("marketScene");
    }

    public void LoadDnE()
    {
        TransitionTo("dailyneventScene");
    }

    public void LoadInventory()
    {
        TransitionTo("inventoryScene");
    }

    public void LoadAIScript()
    {
        TransitionTo("AIScriptScene");
    }

    public void LoadAvatar()
    {
        TransitionTo("avatarScene");
    }

    public void LoadLoadout()
    {
        TransitionTo("loadoutScene");
    }

    public void LoadProfile()
    {
        TransitionTo("profileScene");
    }

    public void LoadSettings()
    {
        TransitionTo("settingsScene");
    }

    public void LoadLeaderboad()
    {
        TransitionTo("leaderboardScene");
    }

    //STAGE SCENE BATTLE

    //Campaign Mode
    public void LoadCampaign()
    {
        TransitionTo("campaignScene");
    }

    //Multiplayer Mode
    public void LoadMulti()
    {
        TransitionTo("multiScene");
    }

    //Battle Tutorial
    public void LoadTutorial1()
    {
        TransitionTo("tutorial1Scene");
    }

    public void LoadTutorial2()
    {
        TransitionTo("tutorial2Scene");
    }

    public void LoadTutorial3()
    {
        TransitionTo("tutorial3Scene");
    }

    public void LoadTutorial4()
    {
        TransitionTo("tutorial4Scene");
    }

    //Battle Campaign 1
    public void LoadC1_1()
    {
        TransitionTo("C1_1Scene");
    }

    public void LoadC1_2()
    {
        TransitionTo("C1_2Scene");
    }

    public void LoadC1_3()
    {
        TransitionTo("C1_3Scene");
    }

    public void LoadC1_4()
    {
        TransitionTo("C1_4Scene");
    }

    public void LoadC1_5()
    {
        TransitionTo("C1_5Scene");
    }

    //Battle Campaign 2
    public void LoadC2_1()
    {
        TransitionTo("C2_1Scene");
    }

    public void LoadC2_2()
    {
        TransitionTo("C2_2Scene");
    }

    public void LoadC2_3()
    {
        TransitionTo("C2_3Scene");
    }

    public void LoadC2_4()
    {
        TransitionTo("C2_4Scene");
    }

    public void LoadC2_5()
    {
        TransitionTo("C2_5Scene");
    }

    //Battle Campaign 3
    public void LoadC3_1()
    {
        TransitionTo("C3_1Scene");
    }

    public void LoadC3_2()
    {
        TransitionTo("C3_2Scene");
    }

    public void LoadC3_3()
    {
        TransitionTo("C3_3Scene");
    }

    public void LoadC3_4()
    {
        TransitionTo("C3_4Scene");
    }

    public void LoadC3_5()
    {
        TransitionTo("C3_5Scene");
    }

    //Battle Campaign 4
    public void LoadC4_1()
    {
        TransitionTo("C4_1Scene");
    }

    public void LoadC4_2()
    {
        TransitionTo("C4_2Scene");
    }

    public void LoadC4_3()
    {
        TransitionTo("C4_3Scene");
    }

    public void LoadC4_4()
    {
        TransitionTo("C4_4Scene");
    }

    public void LoadC4_5()
    {
        TransitionTo("C4_5Scene");
    }

    // Tombol untuk keluar dari game
    public void ExitGame()
    {
        Debug.Log("Keluar dari game...");
        Application.Quit();
    }

#if UNITY_STANDALONE_WIN
    [DllImport("user32.dll")]
    private static extern bool ShowWindow(System.IntPtr hWnd, int nCmdShow);

    [DllImport("user32.dll")]
    private static extern System.IntPtr GetActiveWindow();

    private const int SW_MINIMIZE = 6;
#endif

    public void MinimizeGame()
    {
#if UNITY_STANDALONE_WIN
        var hwnd = GetActiveWindow();
        ShowWindow(hwnd, SW_MINIMIZE);
        Debug.Log("Game diminimize");
#else
        Debug.LogWarning("Fitur minimize cuma ada di Windows Standalone build.");
#endif
    }

}
