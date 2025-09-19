using SumoCore;
using UnityEngine;
using UnityEngine.UI;

public class PartSwitcher : MonoBehaviour
{
    public SumoPart part;
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
        SFXManager.Instance.Play2D("ui_accept_small");
        if (GameManager.Instance.GetProfileById() == null)
            return;

        if (sprites.Length == 0 || targetImage == null)
            return;

        currentIndex = (currentIndex + direction + sprites.Length) % sprites.Length;
        targetImage.sprite = sprites[currentIndex];
        targetPreviewImage.sprite = sprites[currentIndex];

        var costume = GameManager.Instance.GetProfileById();
        costume.Parts[part] = sprites[currentIndex];
    }
}
