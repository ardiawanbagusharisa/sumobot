using UnityEngine.SceneManagement;

// Cross-scene navigation handoff for the Workshop pillar (decision-10, E3.2). The My Creations
// list lives as a panel inside the MainMenu scene, but the node editor is its own scene, so the
// two need a small amount of state that survives a scene load:
//   - forward:  which draft the editor should open (OpenEditor)
//   - backward: a flag asking MainMenu to re-open the Workshop panel instead of Login (ExitToWorkshop)
//
// Mirrors the GameManager.EditingID pattern (a static field carried across LoadScene) but scoped
// to the Workshop so GameManager does not accumulate unrelated routing. Depends only on
// SceneManager — no BotGraph or service coupling. All state is consume-once so nothing leaks into
// a later, unrelated navigation.
public static class WorkshopRouting
{
    public const string EditorSceneName = "BotGraphEditor";
    public const string MainMenuSceneName = "MainMenu";

    /// <summary>The draft the editor should open, or null to start a brand-new bot (which is
    /// persisted only when the author hits Save). Set by <see cref="OpenEditor"/>, read once by
    /// the editor on load.</summary>
    public static string PendingDraftId { get; private set; }

    private static bool returnToWorkshop;

    /// <summary>Enter the node editor. Pass an existing draft id to edit it, or nothing/null to
    /// start a new bot.</summary>
    public static void OpenEditor(string draftId = null)
    {
        PendingDraftId = draftId;
        SceneManager.LoadScene(EditorSceneName);
    }

    /// <summary>Leave the editor back to the MainMenu, asking it to land on the Workshop panel
    /// rather than the default Login screen.</summary>
    public static void ExitToWorkshop()
    {
        returnToWorkshop = true;
        PendingDraftId = null;
        SceneManager.LoadScene(MainMenuSceneName);
    }

    /// <summary>Consume-once: MainMenuController asks on Start whether it should open the Workshop
    /// panel (true only when we just came back from the editor). Reading it clears it.</summary>
    public static bool ConsumeReturnToWorkshop()
    {
        bool value = returnToWorkshop;
        returnToWorkshop = false;
        return value;
    }
}
