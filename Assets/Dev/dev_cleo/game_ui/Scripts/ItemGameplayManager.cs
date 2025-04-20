using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ItemGameplayManager : MonoBehaviour
{
    [System.Serializable]
    public class RobotData
    {
        public string robotName;
        public Sprite robotSprite;
        public string abilityText;
        public string statText;
    }

    public Image robotDisplayImage;
    public TMP_Text abilityText;
    public TMP_Text statText;

    public RobotData[] robots;

    // Fungsi dipanggil tiap tombol ditekan
    public void SelectRobot(int index)
    {
        if (index < 0 || index >= robots.Length) return;

        robotDisplayImage.sprite = robots[index].robotSprite;
        abilityText.text = robots[index].abilityText;
        statText.text = robots[index].statText;
    }
}
