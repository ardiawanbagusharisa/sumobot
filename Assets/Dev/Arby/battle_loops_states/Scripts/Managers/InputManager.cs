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

    public void RegisterInput(GameObject player, BattleInputType inputType)
    {
        battleInputType = inputType;

        var playerIdx = player.GetComponent<SumoRobot>().IdInt;
        var input = new SumoRobotInput();

        input.Id = playerIdx;
        input.sumo = player.GetComponent<SumoRobotController>();
        InputPlayers[playerIdx] = input;

        PrepareInput(playerIdx);
    }
    public void UnregisterInput(GameObject player, BattleInputType inputType)
    {
        var playerIdx = player.GetComponent<SumoRobot>().IdInt;
        battleInputType = inputType;

        InputPlayers.TryGetValue(playerIdx, out SumoRobotInput sumoInput);
        if (sumoInput != null)
        {
            Destroy(sumoInput.spawnedUIInput);
            InputPlayers.Remove(playerIdx);
        }
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
            InputPlayers[playerIdx].spawnedUIInput = spawnedLiveCommand;

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
            InputPlayers[playerIdx].spawnedUIInput = spawnedLiveCommand;

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
            InputPlayers[playerIdx].spawnedUIInput = spawnedLeftButtons;
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

            InputPlayers[playerIdx].spawnedUIInput = spawnedRightButtons;
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
    public GameObject spawnedUIInput;
}