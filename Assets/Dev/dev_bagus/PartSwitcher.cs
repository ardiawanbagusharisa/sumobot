using UnityEngine;
using UnityEngine.UI;

public class PartSwitcher : MonoBehaviour
{
    public Sprite[] sprites;     
    public Image targetImage;
    public Image targetPreviewImage;

    private int currentIndex = 0;

    void Start()
    {
        if (sprites.Length > 0 && targetImage != null)
        {
            targetImage.sprite = sprites[currentIndex];
            targetPreviewImage.sprite = sprites[currentIndex];
        }
    }

    public void UpdateSprite(int direction)
    {
        if (sprites.Length == 0 || targetImage == null) return;

        currentIndex = (currentIndex + direction + sprites.Length) % sprites.Length;
        targetImage.sprite = sprites[currentIndex];
        targetPreviewImage.sprite = sprites[currentIndex];
    }
}
