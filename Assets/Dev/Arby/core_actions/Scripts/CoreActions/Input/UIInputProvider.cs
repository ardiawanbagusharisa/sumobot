using System.Collections.Generic;
using UnityEngine;
using RobotCoreAction.Inputs;

namespace RobotCoreAction
{
    namespace Inputs
    {
        public class UIInputProvider : MonoBehaviour, IInputProvider
        {
            public bool IsEnabled { get; private set; }

            private void OnEnable()
            {
                IsEnabled = true;
            }
            void OnDisable()
            {
                IsEnabled = false;
            }

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
}