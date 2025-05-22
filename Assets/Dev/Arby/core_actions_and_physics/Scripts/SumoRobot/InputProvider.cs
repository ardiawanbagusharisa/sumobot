using System;
using System.Collections.Generic;
using BattleLoop;
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
        public Dictionary<string, bool> StateKeyboardAction;
        public Dictionary<PlayerSide, Dictionary<KeyCode, ISumoAction>> KeyboardBindings;

        private Queue<ISumoAction> CommandQueue = new Queue<ISumoAction>();


        public InputProvider(PlayerSide side, bool includeKeyboard = false)
        {
            PlayerSide = side;
            IncludeKeyboard = includeKeyboard;
        }

        void OnEnable()
        {
            StateKeyboardAction = new Dictionary<string, bool>()
                                    {
                                        {"AccelerateAction",true},
                                        {"TurnRightAction",true},
                                        {"TurnLeftAction",true},
                                        {"DashAction",true},
                                        {"SkillAction",true},
                                    };

            KeyboardBindings = new Dictionary<PlayerSide, Dictionary<KeyCode, ISumoAction>>()
                            {
                                {PlayerSide.Left, new Dictionary<KeyCode, ISumoAction>(){
                                    { KeyCode.W, new AccelerateAction(InputType.Keyboard) },
                                    { KeyCode.D, new TurnRightAction(InputType.Keyboard) },
                                    { KeyCode.A, new TurnLeftAction(InputType.Keyboard)},
                                    { KeyCode.LeftShift, new DashAction(InputType.Keyboard)},
                                    { KeyCode.C, new SkillAction(SkillType,InputType.Keyboard)},
                                }},
                                {PlayerSide.Right, new Dictionary<KeyCode,ISumoAction>(){
                                    { KeyCode.O, new AccelerateAction(InputType.Keyboard)},
                                    { KeyCode.Semicolon, new TurnRightAction(InputType.Keyboard)},
                                    { KeyCode.K, new TurnLeftAction(InputType.Keyboard)},
                                    { KeyCode.RightShift, new DashAction(InputType.Keyboard)},
                                    { KeyCode.M, new SkillAction(SkillType,InputType.Keyboard)},
                                }},
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

            Dictionary<KeyCode, ISumoAction> sideKeyboard = KeyboardBindings[PlayerSide];
            foreach (var item in sideKeyboard)
            {
                // Map input to actions
                if (Input.GetKey(item.Key) && StateKeyboardAction[item.Value.GetType().Name])
                {
                    actions.Add(item.Value);
                }

                // // Log started when player press & hold button
                // if (Input.GetKeyDown(item.Key) && StateKeyboardAction[item.Value.GetType().Name] && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
                // {
                //     LogManager.LogRoundEvent(

                //         actor: PlayerSide.ToLogActorType(),
                //         detail: new Dictionary<string, object>()
                //                 {
                //                     {"action", item.Value.GetType().Name},
                //                     {"active", true},
                //                 });
                // }

                // // Log ended when player release button
                // if (Input.GetKeyUp(item.Key) && StateKeyboardAction[item.Value.GetType().Name] && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
                // {
                //     LogManager.LogRoundEvent(
                //         actor: PlayerSide.ToLogActorType(),
                //         detail: new Dictionary<string, object>()
                //                 {
                //                     {"action", item.Value.GetType().Name},
                //                     {"active", false},
                //                 });
                // }
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
