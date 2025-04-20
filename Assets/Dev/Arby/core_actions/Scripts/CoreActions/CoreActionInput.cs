using System.Collections.Generic;
using UnityEngine;

namespace RobotCoreAction
{


    public interface IInputProvider
    {
        List<ISumoAction> GetInput();
    }

    public class KeyboardInputProvider : IInputProvider
    {
        public List<ISumoAction> GetInput()
        {
            var actions = new List<ISumoAction>();

            if (Input.GetAxis("Vertical") > 0)
                actions.Add(new AccelerateAction());
            if (Input.GetKeyUp(KeyCode.Space))
                actions.Add(new DashAction());
            if (Input.GetKey(KeyCode.RightArrow))
                actions.Add(new TurnAction(true));
            if (Input.GetKey(KeyCode.LeftArrow))
                actions.Add(new TurnAction(false));
            if (Input.GetKeyUp(KeyCode.Slash))
                actions.Add(new SkillAction(new StoneSkill()));
            if (Input.GetKeyUp(KeyCode.RightShift))
                actions.Add(new SkillAction(new BoostSkill()));

            return actions;
        }
    }

    public class ScriptInputProvider : IInputProvider
    {

        private Queue<ISumoAction> commandQueue = new Queue<ISumoAction>();

        // This will be called in AI Submission / Live Command
        public void EnqueueCommand(ISumoAction action)
        {
            commandQueue.Enqueue(action);
        }

        public List<ISumoAction> GetInput()
        {
            var actions = new List<ISumoAction>();
            while (commandQueue.Count > 0)
                actions.Add(commandQueue.Dequeue());
            return actions;
        }
    }

    public class UIInputProvider : IInputProvider
    {

        private Queue<ISumoAction> commandQueue = new Queue<ISumoAction>();
        public void OnAccelerateButtonPressed()
        {
            commandQueue.Enqueue(new AccelerateAction());
        }
        public void OnDashButtonPressed()
        {
            commandQueue.Enqueue(new DashAction());
        }

        public void OnTurnButtonPressed(bool isRight)
        {
            commandQueue.Enqueue(new TurnAction(isRight));
        }

        public void OnSkillsPressed(ISkill skill)
        {
            commandQueue.Enqueue(new SkillAction(skill));
        }

        public List<ISumoAction> GetInput()
        {
            var actions = new List<ISumoAction>();
            while (commandQueue.Count > 0)
                actions.Add(commandQueue.Dequeue());
            return actions;
        }

    }

}
