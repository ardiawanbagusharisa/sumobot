using System;
using System.Collections.Generic;
using System.Linq;
using CoreSumo;
using UnityEngine;

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
    // AccelerateAction: true, means player can press Accelerate
    public Dictionary<string, bool> StateKeyboardAction;

    // Store keyboard configurations (keybindings)
    public static readonly Dictionary<PlayerSide, Dictionary<KeyCode, ISumoAction>> KeyboardBindings
        = new Dictionary<PlayerSide, Dictionary<KeyCode, ISumoAction>>()
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

    private Queue<ISumoAction> commandQueue = new Queue<ISumoAction>();
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
            StateKeyboardAction.Add(action.ToString(), true);
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
        var actions = new List<ISumoAction>();

        Dictionary<KeyCode, ISumoAction> sideKeyboard = KeyboardBindings[PlayerSide];
        foreach (var item in sideKeyboard)
        {
            // Map input to actions
            if (Input.GetKey(item.Key) && StateKeyboardAction[item.Value.Type.ToString()])
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
    public void OnAccelerateButtonPressed()
    {
        EnqueueCommand(new AccelerateAction(InputType.UI));
    }

    public void OnDashButtonPressed()
    {
        EnqueueCommand(new DashAction(InputType.UI));
    }

    public void OnTurnLeftButtonPressed()
    {
        EnqueueCommand(new TurnAction(InputType.UI, ActionType.TurnLeft));
    }

    public void OnTurnRightButtonPressed()
    {
        EnqueueCommand(new TurnAction(InputType.UI, ActionType.TurnRight));
    }

    public void OnBoostSkillButtonPressed()
    {
        EnqueueCommand(new SkillAction(InputType.UI));
    }

    public void OnStoneSkillButtonPressed()
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

        if (action is TurnAction)
        {
            if (action.Type == ActionType.TurnLeftWithAngle || action.Type == ActionType.TurnRightWithAngle)
            {
                float param = (float)action.Param;
                float minAngle = controller.HalfTurnAngle.min;
                float maxAngle = controller.HalfTurnAngle.max;
                if (param < minAngle || param > maxAngle)
                    throw new Exception($"parameter can't be < {minAngle} or > {maxAngle} when you are using [{action.FullName}]");
            }
            else if (action.Type == ActionType.TurnWithAngle)
            {
                float param = (float)action.Param;
                float minAngle = controller.FullTurnAngle.min;
                float maxAngle = controller.FullTurnAngle.max;
                if (param < minAngle || param > maxAngle)
                    throw new Exception($"param can't be < {minAngle} and > {maxAngle} when you are using [${action.FullName}] type");
            }
        }
        return true;
    }
    #endregion
}
