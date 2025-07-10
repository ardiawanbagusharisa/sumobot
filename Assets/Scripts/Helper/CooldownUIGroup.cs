using System.Linq;
using SumoInput;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class CooldownUIGroup : MonoBehaviour
{
    private Image fillImage;
    private TMP_Text text;

    private void OnEnable()
    {
        fillImage = transform.GetComponentsInChildren<Image>().ToList().FirstOrDefault((x) => x.gameObject != gameObject);
        text = transform.GetComponentsInChildren<TMP_Text>().ToList().FirstOrDefault((x) => x.gameObject != gameObject);
    }

    public void SetCooldown(float normalized)
    {
        if (fillImage != null)
        {
            fillImage.fillAmount = normalized;
        }
    }

    public void SetText(string text)
    {
        if (this.text != null)
            this.text.SetText(text);
    }

    public void SetVisible(bool visible)
    {
        gameObject.SetActive(visible);
    }
}

[System.Serializable]
public class CooldownUIGroupSet
{
    public CooldownUIGroup buttonUI;  // for InputType.UI (Keyboard)
    public CooldownUIGroup globalUI;  // for gamepad/AI

    public void SetCooldown(float normalized, InputType inputType)
    {
        Active(inputType)?.SetCooldown(normalized);
    }

    public void SetVisible(InputType inputType)
    {
        if (inputType == InputType.UI || inputType == InputType.Keyboard)
        {
            buttonUI.SetVisible(true);
            globalUI.SetVisible(false);
        }
        else
        {
            buttonUI.SetVisible(false);
            globalUI.SetVisible(true);
        }
    }

    private CooldownUIGroup Active(InputType inputType)
    {
        return (inputType == InputType.UI || inputType == InputType.Keyboard) ? buttonUI : globalUI;
    }

    public void Reset(InputType inputType)
    {
        SetCooldown(0, inputType);
    }

    public void SetText(string value)
    {
        globalUI.SetText(value);
        buttonUI.SetText(value);
    }
}
