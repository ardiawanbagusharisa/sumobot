using System;
using System.Reflection;
using SumoCampaign;
using SumoCore;
using SumoInput;
using SumoManager;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

[InitializeOnLoad]
public static class CampaignRuntimeSmoke
{
    private const string ActiveKey = "sumobot.campaign.smoke.active";
    private const string StageKey = "sumobot.campaign.smoke.stage";
    private const string StartedKey = "sumobot.campaign.smoke.started";
    private const string LevelIndexKey = "sumobot.campaign.smoke.levelIndex";

    static CampaignRuntimeSmoke()
    {
        EditorApplication.update += Tick;
    }

    public static void Run()
    {
        SessionState.SetBool(ActiveKey, true);
        SessionState.SetInt(StageKey, 0);
        SessionState.SetInt(LevelIndexKey, 0);
        SessionState.SetFloat(StartedKey, (float)EditorApplication.timeSinceStartup);
        EditorSceneManager.OpenScene("Assets/Scenes/Campaign_1_1.unity");
        EditorApplication.isPlaying = true;
    }

    private static void Next(int stage)
    {
        SessionState.SetInt(StageKey, stage);
        SessionState.SetFloat(StartedKey, (float)EditorApplication.timeSinceStartup);
    }

    private static void Tick()
    {
        if (!SessionState.GetBool(ActiveKey, false) || !EditorApplication.isPlaying)
            return;
        try
        {
            int stage = SessionState.GetInt(StageKey, 0);
            double elapsed = EditorApplication.timeSinceStartup - SessionState.GetFloat(StartedKey, 0f);
            if (elapsed > 28f) throw new Exception("Timed out at stage " + stage);
            var controller = UnityEngine.Object.FindFirstObjectByType<CampaignLevelController>();
            var missionView = UnityEngine.Object.FindFirstObjectByType<CampaignMissionView>(
                FindObjectsInactive.Include);
            switch (stage)
            {
                case 0:
                    if (controller == null || missionView == null || !missionView.Panel.activeSelf ||
                        controller.Level == null ||
                        controller.GetComponent<BattleManager>().Battle.LeftPlayer == null) return;
                    Require(controller.Level.Id == "1-1", "Wrong opening level.");
                    controller.StartMission();
                    Next(1);
                    break;
                case 1:
                    if (controller.GetComponent<BattleManager>().CurrentState != BattleState.Battle_Ongoing)
                        return;
                    var left = controller.GetComponent<BattleManager>().Battle.LeftPlayer;
                    var right = controller.GetComponent<BattleManager>().Battle.RightPlayer;
                    Require(left.InputProvider != null && left.InputProvider.IncludeKeyboard,
                        "Buttons lesson did not enable left keyboard input.");
                    Require(right.InputProvider == null || !right.InputProvider.IncludeKeyboard,
                        "Campaign opponent accepted keyboard input.");
                    left.InputProvider.EnqueueCommand(new AccelerateAction(InputType.Keyboard, 1.5f));
                    Next(2);
                    break;
                case 2:
                    if (!PrivateBool(controller, "actionMove")) return;
                    left = controller.GetComponent<BattleManager>().Battle.LeftPlayer;
                    left.transform.position = left.StartPosition + Vector3.right * 2.2f;
                    Next(3);
                    break;
                case 3:
                    if (!PrivateBool(controller, "successPending")) return;
                    Require(missionView.CompletionSticker.activeSelf, "Success sticker did not appear immediately.");
                    Require(!missionView.Panel.activeSelf, "Success panel appeared before follow-through.");
                    Next(4);
                    break;
                case 4:
                    if (elapsed < 3f)
                    {
                        Require(missionView.CompletionSticker.activeSelf && !missionView.Panel.activeSelf,
                            "Success did not remain visible during the delay.");
                        return;
                    }
                    if (elapsed < 5.2f) return;
                    Require(!missionView.CompletionSticker.activeSelf && missionView.Panel.activeSelf &&
                        PrivateBool(controller, "finished"),
                        "Success panel did not replace the sticker after five seconds.");
                    Time.timeScale = 1f;
                    SceneManager.LoadScene("MainMenu");
                    Next(5);
                    break;
                case 5:
                    if (elapsed < 1f) return;
                    Require(UnityEngine.Object.FindObjectsByType<BGMManager>(FindObjectsSortMode.None).Length == 1,
                        "BGM manager duplicated on return to menu.");
                    Require(UnityEngine.Object.FindObjectsByType<SFXManager>(FindObjectsSortMode.None).Length == 1,
                        "SFX manager duplicated on return to menu.");
                    CampaignCatalog.Launch(CampaignCatalog.Levels[5]);
                    Next(6);
                    break;
                case 6:
                    if (controller == null || missionView == null || !missionView.Panel.activeSelf ||
                        controller.Level.Id != "2-1" ||
                        controller.GetComponent<BattleManager>().Battle.LeftPlayer == null) return;
                    controller.StartMission();
                    Next(7);
                    break;
                case 7:
                    if (controller.GetComponent<BattleManager>().CurrentState != BattleState.Battle_Ongoing)
                        return;
                    Require(!controller.GetComponent<BattleManager>().Battle.LeftPlayer.InputProvider.IncludeKeyboard,
                        "Live Command lesson accepted keyboard movement.");
                    CampaignCatalog.Launch(CampaignCatalog.Levels[10]);
                    Next(8);
                    break;
                case 8:
                    if (controller == null || missionView == null || !missionView.Panel.activeSelf ||
                        controller.Level.Id != "3-1" ||
                        controller.GetComponent<BattleManager>().Battle.LeftPlayer == null) return;
                    controller.StartMission();
                    Next(9);
                    break;
                case 9:
                    if (controller.GetComponent<BattleManager>().CurrentState != BattleState.Battle_Ongoing)
                        return;
                    Require(!controller.GetComponent<BattleManager>().Battle.LeftPlayer.InputProvider.IncludeKeyboard,
                        "FSM lesson accepted keyboard movement.");
                    CampaignCatalog.Launch(CampaignCatalog.Levels[0]);
                    Next(10);
                    break;
                case 10:
                    int index = SessionState.GetInt(LevelIndexKey, 0);
                    var level = CampaignCatalog.Levels[index];
                    if (controller == null || missionView == null || !missionView.Panel.activeSelf ||
                        controller.Level.Id != level.Id) return;
                    controller.StartMission();
                    Next(11);
                    break;
                case 11:
                    index = SessionState.GetInt(LevelIndexKey, 0);
                    level = CampaignCatalog.Levels[index];
                    var battle = controller.GetComponent<BattleManager>();
                    if (battle.CurrentState != BattleState.Battle_Ongoing) return;
                    Require(Mathf.Abs(battle.ArenaRadius - level.ArenaRadius) < .01f,
                        "Arena radius mismatch in " + level.Id);
                    var circle = battle.Arena.GetComponent<CircleCollider2D>();
                    Require(Mathf.Abs(circle.radius * Mathf.Abs(battle.Arena.transform.lossyScale.x) -
                        level.ArenaRadius) < .01f, "Arena collider mismatch in " + level.Id);
                    Vector2 center = battle.Arena.transform.TransformPoint(circle.offset);
                    Require(Mathf.Abs(Vector2.Distance(battle.Battle.LeftPlayer.StartPosition, center) -
                        level.ArenaRadius * level.SpawnFraction) < .05f,
                        "Player spawn mismatch in " + level.Id);
                    Require(Mathf.Abs(Vector2.Distance(battle.Battle.RightPlayer.StartPosition, center) -
                        level.ArenaRadius * Mathf.Abs(level.OpponentSpawnFraction)) < .05f,
                        "Opponent spawn mismatch in " + level.Id);
                    Require(battle.LeftInputType == level.Input &&
                        battle.Battle.LeftPlayer.InputProvider.IncludeKeyboard ==
                        (level.Input == InputType.UI), "Input mode mismatch in " + level.Id);
                    Require(missionView.CompletionSticker != null &&
                        missionView.CompletionStickerLabel != null,
                        "Mission sticker missing in " + level.Id);
                    Debug.Log("[CampaignRuntimeSmoke] PASS level " + level.Id);
                    index++;
                    SessionState.SetInt(LevelIndexKey, index);
                    if (index == CampaignCatalog.Levels.Count)
                    {
                        Debug.Log("[CampaignRuntimeSmoke] PASS: keyboard action, five-second sticker, audio lifetime, and all 15 runtime level configurations.");
                        SessionState.SetBool(ActiveKey, false);
                        EditorApplication.Exit(0);
                    }
                    else
                    {
                        CampaignCatalog.Launch(CampaignCatalog.Levels[index]);
                        Next(10);
                    }
                    break;
            }
        }
        catch (Exception error)
        {
            Debug.LogError("[CampaignRuntimeSmoke] FAIL: " + error);
            SessionState.SetBool(ActiveKey, false);
            EditorApplication.Exit(1);
        }
    }

    private static bool PrivateBool(object target, string name)
    {
        return (bool)target.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)
            .GetValue(target);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
    }
}
