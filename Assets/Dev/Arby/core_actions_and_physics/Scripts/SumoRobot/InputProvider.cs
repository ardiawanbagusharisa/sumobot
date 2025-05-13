using System;
using System.Collections.Generic;
using UnityEngine;

namespace CoreSumoRobot
{

    public enum InputType
    {
        Keyboard,
        UI,
        LiveCommand,
        Script,
    }


    public class InputProvider : MonoBehaviour
    {
        public bool IncludeKeyboard;
        public PlayerSide PlayerSide;
        public ERobotSkillType SkillType;

        // AccelerateAction: true, means player can press Accelerate
        public Dictionary<string, bool> StateKeyboardAction = new Dictionary<string, bool>(){
            {"AccelerateAction",true},
            {"TurnRightAction",true},
            {"TurnLeftAction",true},
            {"DashAction",true},
            {"SkillAction",true},
        };

        private Queue<ISumoAction> CommandQueue = new Queue<ISumoAction>();


        public InputProvider(PlayerSide side, bool includeKeyboard = false)
        {
            PlayerSide = side;
            IncludeKeyboard = includeKeyboard;
        }

        void OnEnable()
        {
            StateKeyboardAction = new Dictionary<string, bool>(){
            {"AccelerateAction",true},
            {"TurnRightAction",true},
            {"TurnLeftAction",true},
            {"DashAction",true},
            {"SkillAction",true},
        };
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

            if (PlayerSide == PlayerSide.Left)
            {
                if (Input.GetKey(KeyCode.W) && StateKeyboardAction["AccelerateAction"])
                {
                    actions.Add(new AccelerateAction(InputType.Keyboard));
                }
                if (Input.GetKey(KeyCode.D) && StateKeyboardAction["TurnRightAction"])
                {
                    actions.Add(new TurnRightAction(InputType.Keyboard));
                }
                if (Input.GetKey(KeyCode.A) && StateKeyboardAction["TurnRightAction"])
                {
                    actions.Add(new TurnLeftAction(InputType.Keyboard));
                }
                if (Input.GetKeyDown(KeyCode.LeftShift)&& StateKeyboardAction["DashAction"])
                {
                    actions.Add(new DashAction(InputType.Keyboard));
                }
                if (Input.GetKeyDown(KeyCode.C)&& StateKeyboardAction["SkillAction"])
                {
                    actions.Add(new SkillAction(SkillType, InputType.Keyboard));
                }
            }
            else
            {
                if (Input.GetKey(KeyCode.O)&& StateKeyboardAction["AccelerateAction"])
                {
                    actions.Add(new AccelerateAction(InputType.Keyboard));
                }
                if (Input.GetKey(KeyCode.Semicolon)&& StateKeyboardAction["TurnRightAction"])
                {
                    actions.Add(new TurnRightAction(InputType.Keyboard));
                }
                if (Input.GetKey(KeyCode.K)&& StateKeyboardAction["TurnLeftAction"])
                {
                    actions.Add(new TurnLeftAction(InputType.Keyboard));
                }
                if (Input.GetKeyDown(KeyCode.RightShift)&& StateKeyboardAction["DashAction"])
                {
                    actions.Add(new DashAction(InputType.Keyboard));
                }
                if (Input.GetKeyDown(KeyCode.M)&& StateKeyboardAction["SkillAction"])
                {
                    actions.Add(new SkillAction(SkillType, InputType.Keyboard));
                }
            }


            return actions;
        }
        #endregion

        #region UI Input
        public void OnAccelerateButtonPressed()
        {
            CommandQueue.Enqueue(new AccelerateAction(InputType.UI));
        }
        public void OnDashButtonPressed()
        {
            CommandQueue.Enqueue(new DashAction(InputType.UI));
        }

        public void OnTurnLeftButtonPressed()
        {
            CommandQueue.Enqueue(new TurnLeftAction(InputType.UI));
        }

        public void OnTurnRightButtonPressed()
        {
            CommandQueue.Enqueue(new TurnRightAction(InputType.UI));
        }

        public void OnBoostSkillButtonPressed()
        {
            if (SkillType != ERobotSkillType.Boost) return;
            CommandQueue.Enqueue(new SkillAction(ERobotSkillType.Boost, InputType.UI));
        }

        public void OnStoneSkillButtonPressed()
        {
            if (SkillType != ERobotSkillType.Stone) return;
            CommandQueue.Enqueue(new SkillAction(ERobotSkillType.Stone, InputType.UI));
        }
        #endregion
    }

}
