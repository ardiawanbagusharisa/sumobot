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
        Debug.Log($"Example of executing {SumoRobotCommand.PlayerController.IdInt}");
        var script = SumoRobotCommand.inputProvider;
        script.EnqueueCommand(new AccelerateTimeAction(2f));
        script.EnqueueCommand(new TurnAngleAction(180f));
        script.EnqueueCommand(new TurnAngleAction(-90f));

        script.EnqueueCommand(new TurnLeftAngleAction(180f));
        script.EnqueueCommand(new TurnRightAngleAction(90f));

        script.EnqueueCommand(new DashTimeAction(2f));
        script.EnqueueCommand(new SkillAction(ERobotSkillType.Boost));
        script.EnqueueCommand(new SkillAction(ERobotSkillType.Stone));
    }

    #region Example of Gather Info
    public void GetGameplayInfo()
    {
        var timer = BattleManager.Instance.CurrentRound.TimeLeft;
        var rounds = BattleManager.Instance.CurrentRound.RoundNumber;
        var lScore = BattleManager.Instance.Battle.LeftPlayer.Score;
        var rScore = BattleManager.Instance.Battle.RightPlayer.Score;
        BattleWinner winner = BattleManager.Instance.Battle.GetBattleWinner();

        Debug.Log($"timer: {timer}, rounds: {rounds}, leftScore: {lScore}, rightScore: {rScore}, winner: {winner}");
    }

    public void GetRobotsInfo()
    {
        Debug.Log($"LeftSpeed: {BattleManager.Instance.Battle.LeftPlayer.SumoRobotController.DashSpeed}");
        Debug.Log($"ActionsTime: {BattleManager.Instance.Battle.LeftPlayer.SumoRobotController.ActionsTime}");

        Debug.Log($"ActionSkill: {BattleManager.Instance.Battle.LeftPlayer.SumoRobotController.sumoSkill.GetCooldownInfo()}");
        Debug.Log($"ActionSkill: {BattleManager.Instance.Battle.LeftPlayer.SumoRobotController.sumoSkill.IsSkillCooldown()}");
    }

    public void GetPlayerInfo()
    {
        Debug.Log($"LeftSpeed: {BattleManager.Instance.Battle.LeftPlayer.SumoRobotController.IdInt}");
        Debug.Log($"LeftSpeed: {BattleManager.Instance.Battle.LeftPlayer.Score}");
    }

    public void GetLog()
    {
        Battle info = BattleManager.Instance.Battle;
    }
    #endregion
}