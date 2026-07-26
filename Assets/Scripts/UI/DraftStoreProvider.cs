using System.IO;
using SumoBot.Graph.Authoring;
using UnityEngine;

// Single composition point for the bot-draft store on the client. FileDraftStore is engine-free
// (its storage root is injected) so it stays in the pure BotGraph assembly; this is where the UI
// binds it to Application.persistentDataPath. The My Creations list and the editor scene (E3.2)
// share this one instance so a draft saved in the editor shows up in the list.
//
// This is intentionally a small local provider, not a GameServices surface: drafts are local,
// pre-publish work. When publish (E5) turns a draft into a catalog item, that flow can move
// behind GameServices; until then the draft store has no backend to swap.
public static class DraftStoreProvider
{
    private static IDraftStore instance;

    public static IDraftStore Instance =>
        instance ??= new FileDraftStore(Path.Combine(Application.persistentDataPath, "BotDrafts"));
}
