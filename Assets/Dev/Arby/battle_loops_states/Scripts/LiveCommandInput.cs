// implementasi dengan kelas pak bagus (sample)

using BattleLoop;
using CoreSumoRobot;
using UnityEngine;


// Dummy class
public class LiveCommandInput : MonoBehaviour
{
    public SumoRobotController SumoRobotCommand;

    public void Init(SumoRobotController sumoRobotCommand)
    {
        SumoRobotCommand = sumoRobotCommand;
    }

    public void ExampleCommand()
    {
        Debug.Log($"Example of executing {SumoRobotCommand.IdInt}");
        var script = SumoRobotCommand.InputProvider;
        script.EnqueueCommand(new AccelerateTimeAction(2f));
        script.EnqueueCommand(new TurnAngleAction(180f));
        script.EnqueueCommand(new TurnAngleAction(-90f));

        script.EnqueueCommand(new TurnLeftAngleAction(180f));
        script.EnqueueCommand(new TurnRightAngleAction(90f));

        script.EnqueueCommand(new DashTimeAction(2f));
        script.EnqueueCommand(new SkillAction(ERobotSkillType.Boost, InputType.LiveCommand));
        script.EnqueueCommand(new SkillAction(ERobotSkillType.Stone, InputType.LiveCommand));
    }

    #region Example of Gather Info
    public void GetGameplayInfo()
    {
        var timer = BattleManager.Instance.CurrentRound.TimeLeft;
        var rounds = BattleManager.Instance.CurrentRound.RoundNumber;
        // var lScore = BattleManager.Instance.Battle.LeftPlayer.Score;
        // var rScore = BattleManager.Instance.Battle.RightPlayer.Score;
        SumoRobotController winner = BattleManager.Instance.Battle.GetBattleWinner();

        // Debug.Log($"timer: {timer}, rounds: {rounds}, leftScore: {lScore}, rightScore: {rScore}, winner: {winner}");
    }

    public void GetRobotsInfo()
    {
        Debug.Log($"LeftSpeed: {BattleManager.Instance.Battle.LeftPlayer.DashSpeed}");

        // Debug.Log($"ActionSkill: {BattleManager.Instance.Battle.LeftPlayer.Skill.GetCooldownInfo()}");
        Debug.Log($"ActionSkill: {BattleManager.Instance.Battle.LeftPlayer.Skill.SkillCooldown()}");
    }

    public void GetPlayerInfo()
    {
        Debug.Log($"LeftSpeed: {BattleManager.Instance.Battle.LeftPlayer.IdInt}");
        // Debug.Log($"LeftSpeed: {BattleManager.Instance.Battle.LeftPlayer.Score}");
    }

    public void GetLog()
    {
        Battle info = BattleManager.Instance.Battle;
    }
    #endregion
}