using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using RobotCoreAction.Controllers;

namespace RobotCoreAction
{
    namespace Inputs
    {
        public class InputManager : MonoBehaviour
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
                    if (!provider.IsEnabled)
                        continue;
                    var actions = provider.GetInput();
                    foreach (var action in actions)
                    {
                        action.Execute(_robot, _robotStats);
                    }
                }
            }

        }

        public interface IInputProvider
        {
            bool IsEnabled { get; }
            List<ISumoAction> GetInput();
        }

    }

}
