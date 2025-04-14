using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace RobotCoreAction
{
    public class RobotActionInputManager : MonoBehaviour
    {

        private RobotActionController _robot;
        private RobotStats _robotStats; // This will be an object from Robot Stats

        // This will be an object from Input List

        private List<IInputProvider> _providers;

        private void Awake()
        {
            _providers = GetComponents<IInputProvider>().ToList();
            _robot = GetComponent<RobotActionController>();
            _robotStats = GetComponent<RobotStats>();
        }

        private void Update()
        {
            foreach (var provider in _providers)
            {
                var actions = provider.GetInput();
                foreach (var action in actions)
                {
                    action.Execute(_robot, _robotStats);
                }
            }
        }

    }

}
