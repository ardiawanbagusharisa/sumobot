using System;
using System.Collections.Generic;
using System.Data;
using CoreSumoRobot;
using UnityEngine;



public class InputManager : MonoBehaviour
{

    public GameObject LeftButton;
    public GameObject RightButton;

    public GameObject LeftLiveCommand;
    public GameObject RightLiveCommand;

    private BattleInputType battleInputType;
    private Dictionary<int, SumoRobotInput> InputPlayers { get; set; } = new Dictionary<int, SumoRobotInput>();

    public void RegisterInput(GameObject player, BattleInputType inputType)
    {
        battleInputType = inputType;

        SumoRobot playerSumo = player.GetComponent<SumoRobot>();

        SumoRobotInput input = new SumoRobotInput();

        input.Id = playerSumo.IdInt;
        input.sumo = player.GetComponent<SumoRobotController>();
        InputPlayers[playerSumo.IdInt] = input;

        PrepareInput(playerSumo);
    }

    public void UnregisterInput(GameObject player, BattleInputType inputType)
    {
        int playerIdx = player.GetComponent<SumoRobot>().IdInt;
        battleInputType = inputType;

        InputPlayers.TryGetValue(playerIdx, out SumoRobotInput sumoInput);
        if (sumoInput != null)
        {
            InputPlayers.Remove(playerIdx);
        }
    }

    private void PrepareInput(SumoRobot playerSumo)
    {

        // Assigning UI Object to players
        if (playerSumo.IsLeftSide)
        {
            switch (battleInputType)
            {
                case BattleInputType.UI:
                    LeftLiveCommand.SetActive(false);
                    InputPlayers[playerSumo.IdInt].UIInputObject = LeftButton;
                    break;
                case BattleInputType.Script:
                    LeftButton.SetActive(false);
                    InputPlayers[playerSumo.IdInt].UIInputObject = LeftLiveCommand;
                    break;
            }
        }
        else
        {
            switch (battleInputType)
            {
                case BattleInputType.UI:
                    RightLiveCommand.SetActive(false);
                    InputPlayers[playerSumo.IdInt].UIInputObject = RightButton;
                    break;
                case BattleInputType.Script:
                    RightButton.SetActive(false);
                    InputPlayers[playerSumo.IdInt].UIInputObject = RightLiveCommand;
                    break;
            }
        }

        SumoRobotInput input = InputPlayers[playerSumo.IdInt];
        input.UIInputObject.SetActive(true);

        // Declare that Robot uses an input method
        InputProvider inputProvider = input.UIInputObject.GetComponent<InputProvider>();
        input.inputProvider = inputProvider;
        input.sumo.UseInput(inputProvider);


        // Additional initialization
        switch (battleInputType)
        {
            case BattleInputType.UI:
                break;
            case BattleInputType.Script:
                LeftLiveCommand.GetComponent<LiveCommandInput>().Init(input);
                RightLiveCommand.GetComponent<LiveCommandInput>().Init(input);
                break;
        }
    }
}



public class SumoRobotInput
{
    public int Id;
    public SumoRobotController sumo;
    public InputProvider inputProvider;
    public GameObject UIInputObject;
}