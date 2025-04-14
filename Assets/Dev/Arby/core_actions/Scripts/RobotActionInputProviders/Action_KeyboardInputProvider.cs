using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

namespace RobotCoreAction
{
    public class KeyboardInputProvider : MonoBehaviour, IInputProvider
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

            return actions;
        }
    }
}