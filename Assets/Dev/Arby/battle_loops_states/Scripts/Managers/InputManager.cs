using CoreSumoRobot;
using UnityEngine;



public class InputManager : MonoBehaviour
{

    public GameObject LeftButton;
    public GameObject RightButton;

    public GameObject LeftLiveCommand;
    public GameObject RightLiveCommand;

    private BattleInputType battleInputType;

    private SumoRobotInput playerLeftInput;
    private SumoRobotInput playerRightInput;

    public void RegisterInput(GameObject player, BattleInputType inputType)
    {
        battleInputType = inputType;

        SumoRobotInput input = new SumoRobotInput();

        var playerController = player.GetComponent<SumoRobotController>();
        input.PlayerController = playerController;

        if (playerController.Side == PlayerSide.Left)
        {
            playerLeftInput = input;
        }
        else
        {
            playerRightInput = input;
        }

        PrepareInput(playerController);
    }

    public void UnregisterInput(GameObject player, BattleInputType inputType)
    {


        if (player.GetComponent<SumoRobotController>().Side == PlayerSide.Left)
        {
            playerLeftInput = null;
        }
        else
        {
            playerRightInput = null;
        }

    }

    private void PrepareInput(SumoRobotController playerController)
    {
        SumoRobotInput input;

        // Assigning UI Object to players
        if (playerController.Side == PlayerSide.Left)
        {
            switch (battleInputType)
            {
                case BattleInputType.UI:
                    LeftLiveCommand.SetActive(false);
                    playerLeftInput.UIInputObject = LeftButton;
                    break;
                case BattleInputType.LiveCommand:
                    LeftButton.SetActive(false);
                    playerLeftInput.UIInputObject = LeftLiveCommand;
                    break;
            }
            input = playerLeftInput;
        }
        else
        {
            switch (battleInputType)
            {
                case BattleInputType.UI:
                    RightLiveCommand.SetActive(false);
                    playerRightInput.UIInputObject = RightButton;
                    break;
                case BattleInputType.LiveCommand:
                    RightButton.SetActive(false);
                    playerRightInput.UIInputObject = RightLiveCommand;
                    break;
            }
            input = playerRightInput;
        }

        // Declare that Robot uses an input method
        InputProvider inputProvider = input.UIInputObject.GetComponent<InputProvider>();
        input.inputProvider = inputProvider;
        input.PlayerController.UseInput(inputProvider);
        input.UIInputObject.SetActive(true);


        // Additional initialization
        switch (battleInputType)
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
    public InputProvider inputProvider;
    public GameObject UIInputObject;
}