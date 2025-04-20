using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class ItemPreviewManage : MonoBehaviour
{
    [System.Serializable]
    public class ItemData
    {
        public string itemName;
        [TextArea] public string itemDescription;
        public Sprite itemSprite;
        public Button button;
    }

    [Header("Preview Output")]
    public Image previewImage;
    public TMP_Text nameText;
    public TMP_Text descriptionText;

    [Header("Item List")]
    public List<ItemData> items = new List<ItemData>();

    void Start()
    {
        foreach (ItemData item in items)
        {
            item.button.onClick.AddListener(() => UpdatePreview(item));
        }
    }

    void UpdatePreview(ItemData item)
    {
        previewImage.sprite = item.itemSprite;
        nameText.text = item.itemName;
        descriptionText.text = item.itemDescription;
    }
}
