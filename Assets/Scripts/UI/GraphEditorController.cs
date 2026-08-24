using SumoBot.Graph;
using SumoBot.Graph.Authoring;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// The root controller of the node-editor scene (E3.2, decision-10). This is the E3.2 skeleton:
// it establishes the data-bound backbone — a live GraphDocument loaded from / saved to the draft
// store — and the enter/save/exit routing, without yet drawing nodes. Later phases (palette, node
// & port views, wiring, params, live validation) attach to the Document and Library seams this
// exposes; every editor gesture will go through Document so the model stays authoritative.
//
// Attach to a root object in the BotGraphEditor scene and assign the Save / Back buttons (and,
// optionally, a name field). Which draft to open comes from WorkshopRouting.PendingDraftId, set
// before the scene loaded; a null id means "new bot", persisted only on Save.
public class GraphEditorController : MonoBehaviour
{
    [SerializeField] private Button saveButton;          // Save the draft and return to Workshop
    [SerializeField] private Button backButton;          // Return without saving
    [SerializeField] private TMP_InputField nameField;   // Optional: edit the bot's display name

    /// <summary>The live, authoritative model for the graph being edited. Later phases mutate it
    /// through its own methods (AddNode, AddConnection, SetParam, ...).</summary>
    public GraphDocument Document { get; private set; }

    /// <summary>The fixed module vocabulary the palette and validation are built from.</summary>
    public ModuleLibrary Library { get; private set; }

    // The id the draft is saved under. For a new bot this is a fresh id assigned on load, so
    // repeated saves in one session upsert the same draft rather than creating duplicates.
    private string draftId;

    void Awake()
    {
        Library = ModuleLibrary.BuildDefault();
        LoadDocument();

        if (saveButton != null) saveButton.onClick.AddListener(SaveAndExit);
        if (backButton != null) backButton.onClick.AddListener(ExitWithoutSaving);
        if (nameField != null)
        {
            nameField.text = Document.Name ?? "";
            nameField.onValueChanged.AddListener(value => Document.Name = value);
        }
    }

    // Resolve WorkshopRouting.PendingDraftId into a live document: load an existing draft, or start
    // a fresh one. A requested-but-missing draft degrades to a new bot (logged) so a stale id can
    // never leave the editor stuck on an empty scene.
    private void LoadDocument()
    {
        string id = WorkshopRouting.PendingDraftId;

        if (!string.IsNullOrEmpty(id))
        {
            GraphDraft existing = DraftStoreProvider.Instance.Load(id);
            if (existing != null)
            {
                draftId = existing.Id;
                Document = new GraphDocument(Library, existing.Graph ?? new BotGraph());
                return;
            }
            Logger.Error($"[Workshop] Draft '{id}' not found; opening a new bot instead.");
        }

        GraphDraft fresh = GraphDraft.NewDraft(new BotGraph { Name = "New Bot" });
        draftId = fresh.Id;
        Document = new GraphDocument(Library, fresh.Graph);
    }

    /// <summary>Persist the current graph as the draft, then return to the Workshop. Drafts are
    /// work-in-progress, so an invalid graph is still saved (validation surfaces in-editor, E3.2
    /// later phases; only the interpreter refuses to run an invalid graph).</summary>
    public void SaveAndExit()
    {
        Save();
        WorkshopRouting.ExitToWorkshop();
    }

    /// <summary>Upsert the current document into the draft store under its stable id.</summary>
    public void Save() => DraftStoreProvider.Instance.Save(new GraphDraft(draftId, Document.Graph));

    /// <summary>Leave without writing changes back (a new, never-saved bot leaves nothing behind).</summary>
    public void ExitWithoutSaving() => WorkshopRouting.ExitToWorkshop();
}
