using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoManager;
using UnityEngine;

namespace SumoInput
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
        #region Input properties
        public bool IncludeKeyboard;
        public PlayerSide PlayerSide;
        public SumoAPI API;
        #endregion

        #region Runtime properties

        // ActionType.Accelerate -> true, means player can press Accelerate
        public Dictionary<ActionType, bool> StateKeyboardAction;

        // Store keyboard configurations (keybindings)
        public static readonly Dictionary<PlayerSide, Dictionary<KeyCode, ISumoAction>> KeyboardBindings
            = new()
            {
                {PlayerSide.Left, new Dictionary<KeyCode, ISumoAction>(){
                    { KeyCode.W, new AccelerateAction(InputType.Keyboard) },
                    { KeyCode.D, new TurnAction(InputType.Keyboard, ActionType.TurnRight) },
                    { KeyCode.A, new TurnAction(InputType.Keyboard, ActionType.TurnLeft)},
                    { KeyCode.E, new DashAction(InputType.Keyboard)},
                    { KeyCode.Q, new SkillAction(InputType.Keyboard)},
                }},
                {PlayerSide.Right, new Dictionary<KeyCode,ISumoAction>(){
                    { KeyCode.O, new AccelerateAction(InputType.Keyboard)},
                    { KeyCode.Semicolon, new TurnAction(InputType.Keyboard, ActionType.TurnRight)},
                    { KeyCode.K, new TurnAction(InputType.Keyboard, ActionType.TurnLeft)},
                    { KeyCode.RightShift, new DashAction(InputType.Keyboard)},
                    { KeyCode.M, new SkillAction(InputType.Keyboard)},
                }},
            };

        private Queue<ISumoAction> actionQueue = new();
        #endregion

        public InputProvider(PlayerSide side, bool includeKeyboard = false)
        {
            PlayerSide = side;
            IncludeKeyboard = includeKeyboard;
        }

        #region Unity methods
        void OnEnable()
        {
            StateKeyboardAction = new();
            IEnumerable<ActionType> actionTypes = Enum.GetValues(typeof(ActionType)).Cast<ActionType>();
            foreach (ActionType action in actionTypes)
            {
                StateKeyboardAction.Add(action, true);
            }

            actionQueue = new Queue<ISumoAction>();
        }
        #endregion

        #region Input methods

        public List<ISumoAction> FlushAction()
        {
            var result = actionQueue.ToList();
            actionQueue.Clear();
            return result;
        }
    
        // Applied for Live Command And AI Script
        public void EnqueueCommand(ISumoAction action)
        {
            if (IsValid(action))
            {
                actionQueue.Enqueue(action);
            }
        }

        public void EnqueueCommands(Queue<ISumoAction> actions)
        {
            while (actions.Count > 0)
            {
                EnqueueCommand(actions.Dequeue());
            }
        }

        public void ClearCommands()
        {
            actionQueue.Clear();
        }
        #endregion

        #region Keyboard Input
        public void ReadKeyboardInput()
        {
            if (!IncludeKeyboard) return;

            Dictionary<KeyCode, ISumoAction> sideKeyboard = KeyboardBindings[PlayerSide];
            foreach (var item in sideKeyboard)
            {
                if (Input.GetKey(item.Key) && StateKeyboardAction[item.Value.Type])
                {
                    try
                    {
                        EnqueueCommand(item.Value);
                    }
                    catch (Exception e)
                    {
                        Debug.LogError(e);
                    }
                }
            }
        }
        #endregion

        #region UI Input
        public void OnAccelerateButtonPressed(EventParameter _)
        {
            EnqueueCommand(new AccelerateAction(InputType.UI));
        }

        public void OnDashButtonPressed(EventParameter _)
        {
            EnqueueCommand(new DashAction(InputType.UI));
        }

        public void OnTurnLeftButtonPressed(EventParameter _)
        {
            EnqueueCommand(new TurnAction(InputType.UI, ActionType.TurnLeft));
        }

        public void OnTurnRightButtonPressed(EventParameter _)
        {
            EnqueueCommand(new TurnAction(InputType.UI, ActionType.TurnRight));
        }

        public void OnSkillButtonPressed(EventParameter _)
        {
            EnqueueCommand(new SkillAction(InputType.UI));
        }

        public bool IsValid(ISumoAction action)
        {
            if (action is not DashAction && action is not SkillAction)
            {
                float duration = action.Duration;
                if (duration < ISumoAction.MinDuration)
                    throw new Exception($"Duration can't be < {ISumoAction.MinDuration} when you are using [{action.FullName}]");
            }
            return true;
        }

        public bool CanExecute(ISumoAction action)
        {
            Battle battle = BattleManager.Instance.Battle;
            SumoController controller = PlayerSide == PlayerSide.Left ? battle.LeftPlayer : battle.RightPlayer;


            if (action is AccelerateAction)
            {
                if (controller.IsDashActive || controller.IsMovementDisabled)
                    return false;
            }
            if (action is DashAction)
            {
                if (controller.IsDashOnCooldown || controller.IsMovementDisabled)
                    return false;
            }
            if (action is SkillAction)
            {
                if (controller.Skill.IsSkillOnCooldown)
                    return false;
            }
            return true;
        }
        #endregion
    }
}