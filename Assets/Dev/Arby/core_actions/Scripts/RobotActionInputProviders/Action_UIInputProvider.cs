using System.Collections.Generic;
using UnityEngine;

namespace RobotCoreAction
{


    public class UIInputProvider : MonoBehaviour, IInputProvider
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
            commandQueue.Enqueue(new SkillAction());
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