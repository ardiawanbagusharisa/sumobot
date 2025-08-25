
using TMPro;
using UnityEngine;

public class CostumeManager : MonoBehaviour
{
    public TMP_Text PlayerTxt;

    void Start()
    {
        PlayerTxt.text = $"Preview {GameManager.Instance.GetProfileById().Name}'s";
    }

    public void SaveAndBack()
    {
        GameManager.Instance.BotCreator_SaveAndBack();
    }
}