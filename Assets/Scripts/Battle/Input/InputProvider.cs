using System;
using System.Collections.Generic;
using System.Linq;
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
        public SkillType SkillType;
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
                    { KeyCode.LeftShift, new DashAction(InputType.Keyboard)},
                    { KeyCode.C, new SkillAction(InputType.Keyboard)},
                }},
                {PlayerSide.Right, new Dictionary<KeyCode,ISumoAction>(){
                    { KeyCode.O, new AccelerateAction(InputType.Keyboard)},
                    { KeyCode.Semicolon, new TurnAction(InputType.Keyboard, ActionType.TurnRight)},
                    { KeyCode.K, new TurnAction(InputType.Keyboard, ActionType.TurnLeft)},
                    { KeyCode.RightShift, new DashAction(InputType.Keyboard)},
                    { KeyCode.M, new SkillAction(InputType.Keyboard)},
                }},
            };

        private Queue<ISumoAction> commandQueue = new();
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

            commandQueue = new Queue<ISumoAction>();
        }
        #endregion

        #region Input methods
        public List<ISumoAction> GetInput()
        {
            List<ISumoAction> actions = new List<ISumoAction>();

            if (IncludeKeyboard)
                actions = ReadKeyboardInput();

            while (commandQueue.Count > 0)
            {
                actions.Add(commandQueue.Dequeue());
            }

            return actions;
        }
        #endregion

        #region Public API
        // Applied for Live Command And AI Script
        public void EnqueueCommand(ISumoAction action)
        {
            if (IsValid(action))
            {
                commandQueue.Enqueue(action);
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
            commandQueue.Clear();
        }
        #endregion

        #region Keyboard Input
        private List<ISumoAction> ReadKeyboardInput()
        {
            List<ISumoAction> actions = new();

            Dictionary<KeyCode, ISumoAction> sideKeyboard = KeyboardBindings[PlayerSide];
            foreach (var item in sideKeyboard)
            {
                // Map input to actions
                if (Input.GetKey(item.Key) && StateKeyboardAction[item.Value.Type])
                {
                    try
                    {
                        if (IsValid(item.Value))
                            actions.Add(item.Value);
                    }
                    catch (Exception e)
                    {
                        Debug.LogError(e);
                    }
                }
            }
            return actions;
        }
        #endregion

        #region UI Input
        public void OnAccelerateButtonPressed(object[] _)
        {
            EnqueueCommand(new AccelerateAction(InputType.UI));
        }

        public void OnDashButtonPressed(object[] _)
        {
            EnqueueCommand(new DashAction(InputType.UI));
        }

        public void OnTurnLeftButtonPressed(object[] _)
        {
            EnqueueCommand(new TurnAction(InputType.UI, ActionType.TurnLeft));
        }

        public void OnTurnRightButtonPressed(object[] _)
        {
            EnqueueCommand(new TurnAction(InputType.UI, ActionType.TurnRight));
        }

        public void OnSkillButtonPressed(object[] _)
        {
            EnqueueCommand(new SkillAction(InputType.UI));
        }

        public bool IsValid(ISumoAction action)
        {
            Battle battle = BattleManager.Instance.Battle;
            SumoController controller = PlayerSide == PlayerSide.Left ? battle.LeftPlayer : battle.RightPlayer;

            if (action.Param is float)
            {
                float param = (float)action.Param;
                if (param == float.NaN)
                    throw new Exception($"parameter can't be NaN when you are using [{action.FullName}] type");
            }

            if (action.Type == ActionType.TurnLeftWithAngle || action.Type == ActionType.TurnRightWithAngle)
            {
                float param = (float)action.Param;
                float minAngle = controller.HalfTurnAngle.min;
                float maxAngle = controller.HalfTurnAngle.max;
                if (param < minAngle || param > maxAngle)
                    throw new Exception($"parameter can't be < {minAngle} or > {maxAngle} when you are using [{action.FullName}]");
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