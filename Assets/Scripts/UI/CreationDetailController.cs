using System;
using SumoServices;
using SumoBot.Graph.Authoring;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// The Workshop's own detail panel (decision-10) — the authoring-side counterpart to
// ItemDetailController. Duplicate the PanelItemDetail visual, drop the market fields
// (price/creator/win-rate) and swap Ask/Buy for the buttons below. Kept separate from
// ItemDetailController on purpose: a draft is not a CatalogItem, and these actions (edit the
// graph, publish it, delete it) are authoring, not buying.
//
// Two mutually-exclusive button groups by entry kind:
//   - Draft     -> Edit / Publish / Delete   (a published bot is frozen)
//   - Published -> List on Market / Duplicate (immutable; make a new draft to change it)
public class CreationDetailController : MonoBehaviour
{
    [SerializeField] private GameObject panel; // PanelCreationDetail — this controller owns its visibility

    [Header("Detail fields")]
    [SerializeField] private TMP_Text nameText;
    [SerializeField] private Image iconImage;
    [SerializeField] private GameObject descriptionGroup; // hidden when the entry has no description (drafts)
    [SerializeField] private TMP_Text descriptionText;

    [Header("Draft actions")]
    [SerializeField] private Button editButton;
    [SerializeField] private Button publishButton;
    [SerializeField] private Button deleteButton;

    [Header("Published actions")]
    [SerializeField] private Button listMarketButton;
    [SerializeField] private Button duplicateButton;

    [SerializeField] private Button exitButton;

    private CreationEntry current;

    /// <summary>Raised after an action changes the creation set (delete/publish/duplicate) so the
    /// list can refresh.</summary>
    public event Action Changed;

    void Awake()
    {
        if (editButton != null) editButton.onClick.AddListener(OnEditClicked);
        if (publishButton != null) publishButton.onClick.AddListener(OnPublishClicked);
        if (deleteButton != null) deleteButton.onClick.AddListener(OnDeleteClicked);
        if (listMarketButton != null) listMarketButton.onClick.AddListener(OnListMarketClicked);
        if (duplicateButton != null) duplicateButton.onClick.AddListener(OnDuplicateClicked);
        if (exitButton != null) exitButton.onClick.AddListener(Hide);
    }

    public void Show(CreationEntry entry)
    {
        current = entry;
        bool isDraft = entry.Type == CreationEntry.Kind.Draft;

        if (nameText != null) nameText.text = entry.DisplayName;

        if (iconImage != null)
        {
            if (!string.IsNullOrEmpty(entry.IconResourcePath))
                iconImage.sprite = Resources.Load<Sprite>(entry.IconResourcePath);
            iconImage.color = entry.TryGetIconColor(out var tint) ? tint : Color.white;
        }

        // Description exists on published bots only; hide the whole section for drafts (same
        // per-field toggle idea as ItemDetailController.RefreshMetadataFields).
        bool hasDescription = !string.IsNullOrEmpty(entry.Description);
        if (descriptionText != null) descriptionText.text = entry.Description;
        if (descriptionGroup != null) descriptionGroup.SetActive(hasDescription);
        else if (descriptionText != null) descriptionText.gameObject.SetActive(hasDescription);

        if (editButton != null) editButton.gameObject.SetActive(isDraft);
        if (publishButton != null) publishButton.gameObject.SetActive(isDraft);
        if (deleteButton != null) deleteButton.gameObject.SetActive(isDraft);
        if (listMarketButton != null) listMarketButton.gameObject.SetActive(!isDraft);
        if (duplicateButton != null) duplicateButton.gameObject.SetActive(!isDraft);

        if (panel != null)
        {
            SFXManager.Instance.Play2D("ui_accept");
            panel.transform.SetAsLastSibling(); // draw over the list
            panel.SetActive(true);
        }
    }

    public void Hide()
    {
        if (panel == null) return;
        SFXManager.Instance.Play2D("ui_accept");
        panel.SetActive(false);
    }

    // Open the node editor (E3.2) on this draft; it returns to the Workshop on Save/Back.
    private void OnEditClicked()
    {
        if (current == null || current.Type != CreationEntry.Kind.Draft) return;
        WorkshopRouting.OpenEditor(current.Id);
    }

    // Freeze the draft into an owned, sellable bot. This is decision-7 publish (E5,
    // IPublishService), not the market List below. Stubbed until E5.
    private void OnPublishClicked()
    {
        if (current == null || current.Type != CreationEntry.Kind.Draft) return;
        Logger.Warning($"[Workshop] Publish not implemented yet (E5); would publish draft '{current.Id}'.");
        // TODO(E5): await GameServices.Publish.PublishAsync(current.Draft); then Changed?.Invoke();
    }

    private void OnDeleteClicked()
    {
        if (current == null || current.Type != CreationEntry.Kind.Draft) return;
        // TODO: a confirm dialog before this destructive step would be friendlier.
        DraftStoreProvider.Instance.Delete(current.Id);
        Hide();
        Changed?.Invoke();
    }

    // A published bot can be listed on the Market — the same sell flow ItemDetailController owns.
    // Handed off there (or reimplemented) in E5/market wiring; stubbed for now.
    private void OnListMarketClicked()
    {
        if (current == null || current.Type != CreationEntry.Kind.Published) return;
        Logger.Warning($"[Workshop] List-on-Market not wired here yet; item '{current.Id}'.");
        // TODO: reuse the market List flow (GameServices.Market.ListAsync) for current.Published.
    }

    // Published bots are immutable (decision-10); "duplicate" makes a fresh, editable draft from
    // the published graph so the player can iterate. Needs the published item's graph payload
    // (decision-7), which is not surfaced on CatalogItem yet — stubbed until it is.
    private void OnDuplicateClicked()
    {
        if (current == null || current.Type != CreationEntry.Kind.Published) return;
        Logger.Warning($"[Workshop] Duplicate-to-draft not implemented yet; item '{current.Id}'.");
        // TODO(E5): var draft = GraphDraft.NewDraft(graphFrom(current.Published));
        //           DraftStoreProvider.Instance.Save(draft); Changed?.Invoke();
    }
}
