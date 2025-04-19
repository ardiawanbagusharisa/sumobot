using System.Collections.Generic;
using UnityEngine;
using RobotCoreAction.Skills;

namespace RobotCoreAction
{
    namespace Inputs
    {
        public class KeyboardInputProvider : MonoBehaviour, IInputProvider
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

    }
}