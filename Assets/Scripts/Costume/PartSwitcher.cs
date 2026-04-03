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
        if (sprites.Length == 0 || targetImage == null)
            return;

        // Load saved sprite from PlayerProfile if available
        var profile = GameManager.Instance.GetProfileById();
        if (profile != null && profile.Parts.ContainsKey(part) && profile.Parts[part] != null)
        {
            // Find the index of the saved sprite
            Sprite savedSprite = profile.Parts[part];
            for (int i = 0; i < sprites.Length; i++)
            {
                if (sprites[i] == savedSprite)
                {
                    currentIndex = i;
                    break;
                }
            }
        }

        targetImage.sprite = sprites[currentIndex];
        targetPreviewImage.sprite = sprites[currentIndex];
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

        var profile = GameManager.Instance.GetProfileById();
        profile.Parts[part] = sprites[currentIndex];
    }
}
