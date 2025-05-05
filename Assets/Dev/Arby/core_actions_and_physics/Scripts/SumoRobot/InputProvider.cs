using System.Collections.Generic;
using UnityEngine;

namespace CoreSumoRobot
{

    public enum BattleInputType
    {
        Keyboard,
        UI,
        Script
    }


    public class InputProvider : MonoBehaviour
    {
        public bool IncludeKeyboard;
        public bool IsLeftSide;

        private Queue<ISumoAction> CommandQueue = new Queue<ISumoAction>();

        public InputProvider(bool isLeftSide, bool includeKeyboard = false)
        {
            IsLeftSide = isLeftSide;
            IncludeKeyboard = includeKeyboard;
        }

        void Start()
        {
            CommandQueue = new Queue<ISumoAction>();
        }

        public List<ISumoAction> GetInput()
        {
            var actions = new List<ISumoAction>();

            if (IncludeKeyboard)
            {
                actions = ReadKeyboardInput();
            }

            while (CommandQueue.Count > 0)
                actions.Add(CommandQueue.Dequeue());

            return actions;
        }

        #region Live Command / AI Script Input
        // APplied for Live Command And AI Script
        public void EnqueueCommand(ISumoAction action)
        {
            CommandQueue.Enqueue(action);
        }
        #endregion

        #region Keyboard Input
        private List<ISumoAction> ReadKeyboardInput()
        {
            var actions = new List<ISumoAction>();

            if (IsLeftSide)
            {
                if (Input.GetKey(KeyCode.W))
                    actions.Add(new AccelerateAction());
                if (Input.GetKeyUp(KeyCode.LeftShift))
                    actions.Add(new DashAction());
                if (Input.GetKey(KeyCode.D))
                    actions.Add(new TurnRightAction());
                if (Input.GetKey(KeyCode.A))
                    actions.Add(new TurnLeftAction());
                if (Input.GetKeyUp(KeyCode.Q))
                    actions.Add(new SkillAction(ERobotSkillType.Stone));
                if (Input.GetKeyUp(KeyCode.E))
                    actions.Add(new SkillAction(ERobotSkillType.Boost));
            }
            else
            {
                if (Input.GetKey(KeyCode.O))
                    actions.Add(new AccelerateAction());
                if (Input.GetKeyUp(KeyCode.RightShift))
                    actions.Add(new DashAction());
                if (Input.GetKey(KeyCode.Semicolon))
                    actions.Add(new TurnRightAction());
                if (Input.GetKey(KeyCode.K))
                    actions.Add(new TurnLeftAction());
                if (Input.GetKeyUp(KeyCode.I))
                    actions.Add(new SkillAction(ERobotSkillType.Stone));
                if (Input.GetKeyUp(KeyCode.P))
                    actions.Add(new SkillAction(ERobotSkillType.Boost));
            }
            return actions;
        }
        #endregion

        #region UI Input
        public void OnAccelerateButtonPressed()
        {
            CommandQueue.Enqueue(new AccelerateAction());
        }
        public void OnDashButtonPressed()
        {
            CommandQueue.Enqueue(new DashAction());
        }

        public void OnTurnLeft()
        {
            CommandQueue.Enqueue(new TurnLeftAction());
        }

        public void OnTurnRight()
        {
            CommandQueue.Enqueue(new TurnRightAction());
        }

        public void OnBoostSkill()
        {
            CommandQueue.Enqueue(new SkillAction(ERobotSkillType.Boost));
        }

        public void OnStoneSkill()
        {
            CommandQueue.Enqueue(new SkillAction(ERobotSkillType.Stone));
        }
        #endregion
    }

}
