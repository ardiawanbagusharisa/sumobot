using System;
using System.Collections.Generic;
using System.Data;
using CoreSumoRobot;
using UnityEngine;



public class InputManager : MonoBehaviour
{

    public GameObject LiveCmdPrefab;
    public GameObject ButtonPrefab;

    public Transform LeftPanelArea;
    public Transform RightPanelArea;

    private BattleInputType battleInputType;
    private Dictionary<int, SumoRobotInput> InputPlayers { get; set; } = new Dictionary<int, SumoRobotInput>();

    public void RegisterInput(int playerIdx, SumoRobotController sumoRobotController, BattleInputType inputType)
    {
        battleInputType = inputType;

        var sumoCommand = new SumoRobotInput();

        sumoCommand.Id = playerIdx;
        sumoCommand.sumo = sumoRobotController;
        InputPlayers[playerIdx] = sumoCommand;

        PrepareInput(playerIdx);
    }

    private void PrepareInput(int playerIdx)
    {
        switch (battleInputType)
        {
            case BattleInputType.Script:
                SpawnLiveCommandUI(playerIdx);
                break;
            case BattleInputType.UI:
                SpawnButtonUI(playerIdx);
                break;
            case BattleInputType.Keyboard:
                // We dont need to spawn UI object related to the keyboard stuff,
                // so for now it's nothing
                break;
        }
    }

    // Example
    private void SpawnLiveCommandUI(int playerIdx)
    {
        if (playerIdx == 0)
        {
            var spawnedLiveCommand = Instantiate(LiveCmdPrefab, LeftPanelArea.transform);
            spawnedLiveCommand.transform.SetParent(LeftPanelArea.transform, false);

            //Necessary
            spawnedLiveCommand.AddComponent<InputProvider>();
            var inputProvider = spawnedLiveCommand.GetComponent<InputProvider>();
            inputProvider.IsLeftSide = true;
            inputProvider.IncludeKeyboard = false;

            InputPlayers[playerIdx].inputProvider = inputProvider;
            InputPlayers[playerIdx].sumo.UseInput(inputProvider);
            //Necessary

            spawnedLiveCommand.GetComponent<LiveCommandInput>().Init(InputPlayers[playerIdx]);
        }
        else
        {
            var spawnedLiveCommand = Instantiate(LiveCmdPrefab, RightPanelArea.transform);
            spawnedLiveCommand.transform.SetParent(RightPanelArea.transform, false);

            //Necessary
            spawnedLiveCommand.AddComponent<InputProvider>();
            var inputProvider = spawnedLiveCommand.GetComponent<InputProvider>();
            inputProvider.IsLeftSide = false;
            inputProvider.IncludeKeyboard = false;

            InputPlayers[playerIdx].inputProvider = inputProvider;
            InputPlayers[playerIdx].sumo.UseInput(inputProvider);
            //Necessary

            spawnedLiveCommand.GetComponent<LiveCommandInput>().Init(InputPlayers[playerIdx]);
        }
    }

    private void SpawnButtonUI(int playerIdx)
    {
        if (playerIdx == 0)
        {
            var spawnedLeftButtons = Instantiate(ButtonPrefab, LeftPanelArea.transform);
            UIHelper.StretchButtonToParent(spawnedLeftButtons);
            var inputProvider = spawnedLeftButtons.GetComponent<InputProvider>();
            inputProvider.IsLeftSide = true;
            inputProvider.IncludeKeyboard = true;

            InputPlayers[playerIdx].inputProvider = inputProvider;
            InputPlayers[playerIdx].sumo.UseInput(inputProvider);
        }
        else
        {
            var spawnedRightButtons = Instantiate(ButtonPrefab, RightPanelArea.transform);
            UIHelper.StretchButtonToParent(spawnedRightButtons);
            var inputProvider = spawnedRightButtons.GetComponent<InputProvider>();
            inputProvider.IsLeftSide = false;
            inputProvider.IncludeKeyboard = true;

            InputPlayers[playerIdx].inputProvider = inputProvider;
            InputPlayers[playerIdx].sumo.UseInput(inputProvider);
        }

    }
}



public class SumoRobotInput
{
    public int Id;
    public SumoRobotController sumo;
    public IInputProvider inputProvider;
}