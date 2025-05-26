using System;
using BattleLoop;
using CoreSumoRobot;
using UnityEngine;

public class InputManager : MonoBehaviour
{
    public static InputManager Instance { get; private set; }

    public GameObject LeftButton;
    public GameObject RightButton;

    public GameObject LeftLiveCommand;
    public GameObject RightLiveCommand;

    private void Awake()
    {
        if (Instance != null)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;

    }

    public void PrepareInput(SumoRobotController controller)
    {
        GameObject selectedInputObject = null;
        // Assigning UI Object to players
        if (controller.Side == PlayerSide.Left)
        {
            switch (BattleManager.Instance.BattleInputType)
            {

                case InputType.Script:
                    break;
                case InputType.LiveCommand:
                    LeftLiveCommand.SetActive(true);
                    selectedInputObject = LeftLiveCommand;

                    LeftButton.SetActive(false);
                    break;

                // Handle UI And Keyboard
                default:
                    LeftButton.SetActive(true);
                    LeftButton.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                    selectedInputObject = LeftButton;

                    LeftLiveCommand.SetActive(false);
                    break;

            }
        }
        else
        {
            switch (BattleManager.Instance.BattleInputType)
            {
                case InputType.Script:
                    break;
                case InputType.LiveCommand:
                    RightLiveCommand.SetActive(true);
                    selectedInputObject = RightLiveCommand;

                    RightButton.SetActive(false);
                    break;

                // Handle UI And Keyboard
                default:
                    RightButton.SetActive(true);
                    RightButton.GetComponent<ButtonInputHandler>().SetSkillAvailability(controller.Skill.Type);
                    selectedInputObject = RightButton;

                    RightLiveCommand.SetActive(false);
                    break;
            }
        }

        if (selectedInputObject == null)
        {
            throw new Exception("One of [BattleInputType]'s object must be used");
        }

        // Declare that Robot driven by an input provider
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
}