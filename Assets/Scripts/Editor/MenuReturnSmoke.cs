using System;
using System.Reflection;
using System.Threading.Tasks;
using SumoServices;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class MenuReturnSmoke
{
    private const string Active = "sumobot.menu.return.smoke.active";
    private const string Stage = "sumobot.menu.return.smoke.stage";
    private const string Since = "sumobot.menu.return.smoke.since";

    static MenuReturnSmoke() => EditorApplication.update += Tick;

    public static void Run()
    {
        SessionState.SetBool(Active, true);
        Next(0);
        EditorSceneManager.OpenScene("Assets/Scenes/MainMenu.unity");
        EditorApplication.isPlaying = true;
    }

    private static void Next(int stage)
    {
        SessionState.SetInt(Stage, stage);
        SessionState.SetFloat(Since, (float)EditorApplication.timeSinceStartup);
    }

    private static void Tick()
    {
        if (!SessionState.GetBool(Active, false) || !EditorApplication.isPlaying) return;
        try
        {
            int stage = SessionState.GetInt(Stage, 0);
            if (EditorApplication.timeSinceStartup - SessionState.GetFloat(Since, 0f) > 20f)
                throw new Exception("Timed out at stage " + stage);
            var menu = UnityEngine.Object.FindFirstObjectByType<MainMenuController>();
            switch (stage)
            {
                case 0:
                    if (menu == null || !FindMenu("MenuLogin").activeSelf) return;
                    typeof(GameServices).GetProperty(nameof(GameServices.Auth),
                        BindingFlags.Public | BindingFlags.Static)!.SetValue(null, new SignedInTestAuth());
                    menu.ShowCampaign();
                    SceneManager.LoadScene("Battle");
                    Next(1);
                    break;
                case 1:
                    if (SceneManager.GetActiveScene().name != "Battle") return;
                    SceneManager.LoadScene("MainMenu");
                    Next(2);
                    break;
                case 2:
                    if (menu == null || SceneManager.GetActiveScene().name != "MainMenu") return;
                    if (!FindMenu("MenuCampaign").activeSelf || FindMenu("MenuLogin").activeSelf)
                        throw new Exception("Campaign menu was not restored after battle.");
                    Debug.Log("[MenuReturnSmoke] PASS: campaign menu restored after battle without login prompt.");
                    SessionState.SetBool(Active, false);
                    EditorApplication.Exit(0);
                    break;
            }
        }
        catch (Exception error)
        {
            Debug.LogError("[MenuReturnSmoke] FAIL: " + error);
            SessionState.SetBool(Active, false);
            EditorApplication.Exit(1);
        }
    }

    private static GameObject FindMenu(string name)
    {
        foreach (var transform in UnityEngine.Object.FindObjectsByType<Transform>(
                     FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (transform.name == name) return transform.gameObject;
        throw new Exception("Menu missing: " + name);
    }

    private sealed class SignedInTestAuth : IAuthService
    {
        public PlayerAccount Current { get; } = new()
        {
            PlayerId = "menu_return_test", DisplayName = "Tester", IsGuest = false
        };
        public bool IsSignedIn => true;
        public Task<ServiceResult<PlayerAccount>> SignInAnonymouslyAsync() =>
            Task.FromResult(ServiceResult<PlayerAccount>.Ok(Current));
        public Task<ServiceResult<PlayerAccount>> SignUpAsync(string username, string password) =>
            Task.FromResult(ServiceResult<PlayerAccount>.Ok(Current));
        public Task<ServiceResult<PlayerAccount>> SignInAsync(string username, string password) =>
            Task.FromResult(ServiceResult<PlayerAccount>.Ok(Current));
        public Task<ServiceResult<PlayerAccount>> LinkUsernamePasswordAsync(string username, string password) =>
            Task.FromResult(ServiceResult<PlayerAccount>.Ok(Current));
        public Task<ServiceResult> SetDisplayNameAsync(string displayName) =>
            Task.FromResult(ServiceResult.Ok());
        public Task<ServiceResult> SignOutAsync() => Task.FromResult(ServiceResult.Ok());
    }
}
