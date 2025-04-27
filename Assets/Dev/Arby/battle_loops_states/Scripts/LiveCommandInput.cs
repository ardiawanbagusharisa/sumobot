// implementasi dengan kelas pak bagus (sample)

using System.Collections.Generic;
using BattleLoop;
using CoreSumoRobot;
using UnityEngine;


// Dummy class
public class LiveCommandInput : MonoBehaviour
{
    public SumoRobotInput SumoRobotCommand;


    public void Init(SumoRobotInput sumoRobotCommand)
    {
        SumoRobotCommand = sumoRobotCommand;
    }

    public void ExampleCommand()
    {
        Debug.Log($"Example of executing {SumoRobotCommand.Id}");
        var script = (InputProvider)SumoRobotCommand.inputProvider;
        script.EnqueueCommand(new AccelerateTimeAction(2f));
        script.EnqueueCommand(new TurnAngleAction(180f));
        script.EnqueueCommand(new TurnAngleAction(-90f));

        script.EnqueueCommand(new TurnLeftAngleAction(180f));
        script.EnqueueCommand(new TurnRightAngleAction(90f));

        script.EnqueueCommand(new DashTimeAction(2f));
        script.EnqueueCommand(new SkillAction(new StoneSkill()));
        script.EnqueueCommand(new SkillAction(new BoostSkill()));
    }

    #region Example of Gather Info
    public void GetGameplayInfo()
    {
        var timer = BattleManager.Instance.BattleInfo.Time;
        var rounds = BattleManager.Instance.BattleInfo.Rounds;
        var lScore = BattleManager.Instance.BattleInfo.LeftPlayer.Score;
        var rScore = BattleManager.Instance.BattleInfo.RightPlayer.Score;
        BattleWinner winner = BattleManager.Instance.BattleInfo.GetWinner();

        Debug.Log($"timer: {timer}, rounds: {rounds}, leftScore: {lScore}, rightScore: {rScore}, winner: {winner}");
    }

    public void GetRobotsInfo()
    {
        Debug.Log($"LeftSpeed: {BattleManager.Instance.BattleInfo.LeftPlayer.Sumo.DashSpeed}");
        Debug.Log($"ActionsTime: {BattleManager.Instance.BattleInfo.LeftPlayer.SumoRobotController.ActionsTime}");
        Debug.Log($"ActionSkill: {BattleManager.Instance.BattleInfo.LeftPlayer.SumoRobotController.SkillTime}");
    }

    public void GetPlayerInfo()
    {
        Debug.Log($"LeftSpeed: {BattleManager.Instance.BattleInfo.LeftPlayer.Id}");
        Debug.Log($"LeftSpeed: {BattleManager.Instance.BattleInfo.LeftPlayer.Score}");
    }

    public void GetLog()
    {
        Dictionary<string, BattleInfo> infos = BattleManager.Instance.GetLog();
    }
    #endregion
}