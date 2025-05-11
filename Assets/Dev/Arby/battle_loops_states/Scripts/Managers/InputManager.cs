using BattleLoop;
using CoreSumoRobot;
using UnityEngine;

public class InputManager : MonoBehaviour
{
    public GameObject LeftButton;
    public GameObject RightButton;

    public GameObject LeftLiveCommand;
    public GameObject RightLiveCommand;

    private SumoRobotInput playerLeftInput;
    private SumoRobotInput playerRightInput;

    public void InitInput()
    {

        playerLeftInput = new SumoRobotInput(BattleManager.Instance.Battle.LeftPlayer);
        playerRightInput = new SumoRobotInput(BattleManager.Instance.Battle.RightPlayer);

        PrepareInput(playerLeftInput);
        PrepareInput(playerRightInput);
    }

    public void UnInitInput()
    {
        playerLeftInput.PlayerController.UseInput(null);
        playerLeftInput = null;
        playerRightInput.PlayerController.UseInput(null);
        playerRightInput = null;

    }

    private void PrepareInput(SumoRobotInput input)
    {

        // Assigning UI Object to players
        if (input.PlayerController.Side == PlayerSide.Left)
        {
            switch (BattleManager.Instance.BattleInputType)
            {
                case BattleInputType.UI:
                    LeftButton.SetActive(true);
                    LeftLiveCommand.SetActive(false);
                    playerLeftInput.UIInputObject = LeftButton;
                    playerLeftInput.UIInputObject.GetComponent<ButtonInputHandler>().SetSkillAvailability(input.PlayerController.Skill.Type);
                    break;
                case BattleInputType.LiveCommand:
                    LeftLiveCommand.SetActive(true);
                    LeftButton.SetActive(false);
                    playerLeftInput.UIInputObject = LeftLiveCommand;
                    break;
            }
        }
        else
        {
            switch (BattleManager.Instance.BattleInputType)
            {
                case BattleInputType.UI:
                    RightButton.SetActive(true);
                    RightLiveCommand.SetActive(false);
                    playerRightInput.UIInputObject = RightButton;
                    playerRightInput.UIInputObject.GetComponent<ButtonInputHandler>().SetSkillAvailability(input.PlayerController.Skill.Type);
                    break;
                case BattleInputType.LiveCommand:
                    RightLiveCommand.SetActive(true);
                    RightButton.SetActive(false);
                    playerRightInput.UIInputObject = RightLiveCommand;
                    break;
            }
        }

        // Declare that Robot uses an input type
        InputProvider inputProvider = input.UIInputObject.GetComponent<InputProvider>();
        input.InputProvider = inputProvider;
        input.PlayerController.UseInput(inputProvider);

        // Additional initialization
        switch (BattleManager.Instance.BattleInputType)
        {
            case BattleInputType.UI:
                break;
            case BattleInputType.LiveCommand:
                LeftLiveCommand.GetComponent<LiveCommandInput>().Init(input);
                RightLiveCommand.GetComponent<LiveCommandInput>().Init(input);
                break;
        }
    }
}



public class SumoRobotInput
{
    public SumoRobotController PlayerController;
    public InputProvider InputProvider;
    public GameObject UIInputObject;

    public SumoRobotInput(SumoRobotController controller)
    {
        PlayerController = controller;
    }
}