using System;
using CoreSumo;
using UnityEngine;

public class InputManager : MonoBehaviour
{
    public static InputManager Instance { get; private set; }

    #region Input properties
    public GameObject LeftButton;
    public GameObject RightButton;
    public GameObject LeftLiveCommand;
    public GameObject RightLiveCommand;
    #endregion

    #region Unity methods
    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
    }
    #endregion

    #region Input methods
    public void InitializeInput(SumoController controller)
    {
        GameObject selectedInputObject;

        GameObject liveCommandObject = controller.Side == PlayerSide.Left ? LeftLiveCommand : RightLiveCommand;
        GameObject UIButtonsObject = controller.Side == PlayerSide.Left ? LeftButton : RightButton;

        switch (BattleManager.Instance.BattleInputType)
        {
            case InputType.Script:
                liveCommandObject.SetActive(false);
                UIButtonsObject.SetActive(false);
                selectedInputObject = null;
                break;

            case InputType.LiveCommand:
                liveCommandObject.SetActive(true);
                selectedInputObject = liveCommandObject;
                UIButtonsObject.SetActive(false);
                break;

            // UI button & keyboard 
            default:
                UIButtonsObject.SetActive(true);
                UIButtonsObject.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                selectedInputObject = UIButtonsObject;
                liveCommandObject.SetActive(false);
                break;
        }

        if (selectedInputObject == null)
            throw new Exception("One of [BattleInputType]'s object must be used");

        InputProvider inputProvider = selectedInputObject.GetComponent<InputProvider>();
        inputProvider.SkillType = controller.Skill.Type;
        controller.InputProvider = inputProvider;

        // Additional initialization
        switch (BattleManager.Instance.BattleInputType)
        {
            case InputType.UI:
                break;
            case InputType.LiveCommand:
                break;
        }
    }

    public void ResetCooldownButton()
    {
        if (BattleManager.Instance.BattleInputType == InputType.UI || BattleManager.Instance.BattleInputType == InputType.Keyboard)
        {
            LeftButton.GetComponent<ButtonInputHandler>().ResetCooldown();
            RightButton.GetComponent<ButtonInputHandler>().ResetCooldown();
        }
    }
    #endregion
}