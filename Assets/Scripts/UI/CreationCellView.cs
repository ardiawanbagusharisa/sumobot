using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

// View for one My Creations row, instantiated by MyCreationsListController into the grid.
// A plain display card like ItemCellView (name / icon / status badge) — no action buttons; the
// actions live in CreationDetailController, opened on click. The prefab carries two status
// badges, "Draft" and "Publish(ed)"; this toggles the right one by entry kind.
//
// Serialized refs are preferred; if left unassigned they fall back to the child names baked into
// CreationCell.prefab (ItemName / ItemImg / Draft / Publish), so existing instances keep working.
public class CreationCellView : MonoBehaviour
{
    [SerializeField] private TMP_Text nameText;
    [SerializeField] private Image iconImage;
    [SerializeField] private GameObject draftBadge;   // "Draft" label — shown for drafts
    [SerializeField] private GameObject publishBadge; // "Published" label — shown for published bots

    private Button button;

    public CreationEntry Entry { get; private set; }

    /// <summary>Raised when the card is clicked, carrying its bound entry.</summary>
    public event Action<CreationEntry> Clicked;

    void Awake()
    {
        if (nameText == null) nameText = transform.Find("ItemName")?.GetComponent<TMP_Text>();
        if (iconImage == null) iconImage = transform.Find("ItemImg")?.GetComponent<Image>();
        if (draftBadge == null) draftBadge = transform.Find("Draft")?.gameObject;
        if (publishBadge == null) publishBadge = transform.Find("Publish")?.gameObject;

        // The prefab's Button carries a stale persistent onClick copied from ItemCell; replace it
        // with the dynamic click below (same trick as ItemCellView).
        button = GetComponent<Button>();
        button.onClick = new Button.ButtonClickedEvent();
        button.onClick.AddListener(() => Clicked?.Invoke(Entry));
    }

    public void Bind(CreationEntry entry)
    {
        Entry = entry;

        if (nameText != null) nameText.text = entry.DisplayName;

        if (iconImage != null)
        {
            // Published bots carry a catalog icon; drafts keep the prefab's default (the AI icon).
            if (!string.IsNullOrEmpty(entry.IconResourcePath))
                iconImage.sprite = Resources.Load<Sprite>(entry.IconResourcePath);
            iconImage.color = entry.TryGetIconColor(out var tint) ? tint : Color.white;
        }

        bool isDraft = entry.Type == CreationEntry.Kind.Draft;
        if (draftBadge != null) draftBadge.SetActive(isDraft);
        if (publishBadge != null) publishBadge.SetActive(!isDraft);
    }
}
