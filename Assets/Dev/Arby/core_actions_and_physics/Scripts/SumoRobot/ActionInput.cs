using System.Collections.Generic;
using UnityEngine;

namespace CoreSumoRobot
{

    public enum RobotInputType
    {
        Keyboard,
        UI,
        Script
    }
    
    public interface IInputProvider
    {
        List<ISumoAction> GetInput();
    }

    public class KeyboardInputProvider : IInputProvider
    {

        public KeyboardInputProvider(bool isLeftSide)
        {
            this.isLeftSide = isLeftSide;
        }
        private bool isLeftSide;
        public List<ISumoAction> GetInput()
        {
            var actions = new List<ISumoAction>();

            if (isLeftSide)
            {
                if (Input.GetKey(KeyCode.W))
                    actions.Add(new AccelerateAction());
                if (Input.GetKeyUp(KeyCode.LeftShift))
                    actions.Add(new DashAction());
                if (Input.GetKey(KeyCode.D))
                    actions.Add(new TurnAction(true));
                if (Input.GetKey(KeyCode.A))
                    actions.Add(new TurnAction(false));
                if (Input.GetKeyUp(KeyCode.Q))
                    actions.Add(new SkillAction(new StoneSkill()));
                if (Input.GetKeyUp(KeyCode.E))
                    actions.Add(new SkillAction(new BoostSkill()));
            }
            else
            {
                if (Input.GetKey(KeyCode.O))
                    actions.Add(new AccelerateAction());
                if (Input.GetKeyUp(KeyCode.RightShift))
                    actions.Add(new DashAction());
                if (Input.GetKey(KeyCode.Semicolon))
                    actions.Add(new TurnAction(true));
                if (Input.GetKey(KeyCode.K))
                    actions.Add(new TurnAction(false));
                if (Input.GetKeyUp(KeyCode.I))
                    actions.Add(new SkillAction(new StoneSkill()));
                if (Input.GetKeyUp(KeyCode.P))
                    actions.Add(new SkillAction(new BoostSkill()));
            }



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
