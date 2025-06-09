using System;
using System.Collections.Generic;
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
    public bool IncludeKeyboard;
    public PlayerSide PlayerSide;
    public ERobotSkillType SkillType;

    // AccelerateAction: true, means player can press Accelerate
    public Dictionary<string, bool> StateKeyboardAction;

    // Store keyboard configurations (keybindings)
    public static readonly Dictionary<PlayerSide, Dictionary<KeyCode, ISumoAction>> KeyboardBindings
        = new Dictionary<PlayerSide, Dictionary<KeyCode, ISumoAction>>()
                        {
                                {PlayerSide.Left, new Dictionary<KeyCode, ISumoAction>(){
                                    { KeyCode.W, new AccelerateAction(InputType.Keyboard) },
                                    { KeyCode.D, new TurnRightAction(InputType.Keyboard) },
                                    { KeyCode.A, new TurnLeftAction(InputType.Keyboard)},
                                    { KeyCode.LeftShift, new DashAction(InputType.Keyboard)},
                                    { KeyCode.C, new SkillAction(InputType.Keyboard)},
                                }},
                                {PlayerSide.Right, new Dictionary<KeyCode,ISumoAction>(){
                                    { KeyCode.O, new AccelerateAction(InputType.Keyboard)},
                                    { KeyCode.Semicolon, new TurnRightAction(InputType.Keyboard)},
                                    { KeyCode.K, new TurnLeftAction(InputType.Keyboard)},
                                    { KeyCode.RightShift, new DashAction(InputType.Keyboard)},
                                    { KeyCode.M, new SkillAction(InputType.Keyboard)},
                                }},
                        };

    private Queue<ISumoAction> commandQueue = new Queue<ISumoAction>();

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
        commandQueue = new Queue<ISumoAction>();
    }

    public List<ISumoAction> GetInput()
    {
        var actions = new List<ISumoAction>();

        if (IncludeKeyboard)
        {
            actions = ReadKeyboardInput();
        }

        while (commandQueue.Count > 0)
        {
            actions.Add(commandQueue.Dequeue());
        }

        return actions;
    }


    #region Public API
    // APplied for Live Command And AI Script
    public void EnqueueCommand(ISumoAction action)
    {
        Validate(action);
        commandQueue.Enqueue(action);
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
            if (Input.GetKey(item.Key) && StateKeyboardAction[item.Value.GetType().Name])
            {
                actions.Add(item.Value);
            }
        }
        return actions;
    }
    #endregion

    #region UI Input
    public void OnAccelerateButtonPressed()
    {
        commandQueue.Enqueue(new AccelerateAction(InputType.UI));
    }
    
    public void OnDashButtonPressed()
    {
        commandQueue.Enqueue(new DashAction(InputType.UI));
    }

    public void OnTurnLeftButtonPressed()
    {
        commandQueue.Enqueue(new TurnLeftAction(InputType.UI));
    }

    public void OnTurnRightButtonPressed()
    {
        commandQueue.Enqueue(new TurnRightAction(InputType.UI));
    }

    public void OnBoostSkillButtonPressed()
    {
        if (SkillType != ERobotSkillType.Boost) return;
        commandQueue.Enqueue(new SkillAction(InputType.UI));
    }

    public void OnStoneSkillButtonPressed()
    {
        if (SkillType != ERobotSkillType.Stone) return;
        commandQueue.Enqueue(new SkillAction(InputType.UI));
    }

    private void Validate(ISumoAction action)
    {
        SumoController controller = PlayerSide == PlayerSide.Left ? BattleManager.Instance.Battle.LeftPlayer : BattleManager.Instance.Battle.RightPlayer;

        string actionName = action.GetType().Name;

        if (action.Param is float)
        {
            float param = (float)action.Param;
            if (param == float.NaN) throw new Exception($"parameter can't be NaN when you are using [{actionName}] type");
        }

        if (action is TurnLeftAngleAction || action is TurnRightAngleAction)
        {
            float param = (float)action.Param;
            float minAngle = controller.HalfTurnAngle.min;
            float maxAngle = controller.HalfTurnAngle.max;
            if (param < 0) throw new Exception($"parameter can't be < 0 when you are using [{actionName}] type");
            if (param < minAngle || param > maxAngle) throw new Exception($"parameter can't be < {minAngle} and > {maxAngle} when you are using [{actionName}]");
        }
        else if (action is TurnAngleAction)
        {
            float param = (float)action.Param;
            float minAngle = controller.FullTurnAngle.min;
            float maxAngle = controller.FullTurnAngle.max;
            if (param < minAngle || param > maxAngle) throw new Exception($"param can't be < {minAngle} and > {maxAngle} when you are using [${actionName}] type");
        }

        //[Todo] Validate cooldown

    }
    #endregion
}
