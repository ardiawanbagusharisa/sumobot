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
        script.EnqueueCommand(new AccelerateAction());
        script.EnqueueCommand(new DashAction());
        script.EnqueueCommand(new TurnLeftAction());
        script.EnqueueCommand(new AccelerateAction());
    }
}