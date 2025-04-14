using System.Collections.Generic;
using UnityEngine;

namespace RobotCoreAction
{

    public class ScriptInputProvider : MonoBehaviour, IInputProvider
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
}