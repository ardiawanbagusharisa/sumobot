// implementasi dengan kelas pak bagus (sample)

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
}