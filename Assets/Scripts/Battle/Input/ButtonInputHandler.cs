using System;
using System.Collections.Generic;
using CoreSumo;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class ButtonInputHandler : MonoBehaviour
{
    #region UI Lements properties
    public ButtonPointerHandler Accelerate;
    public ButtonPointerHandler TurnLeft;
    public ButtonPointerHandler TurnRight;
    public ButtonPointerHandler Dash;
    public ButtonPointerHandler Stone;
    public ButtonPointerHandler Boost;
    private InputProvider inputProvider;
    #endregion

    #region Runtime properties
    private float delayHoldSeconds = 0.08f;
    private Dictionary<string, Tuple<GameObject, float, InputType?>> lastInputType = new Dictionary<string, Tuple<GameObject, float, InputType?>>();
    #endregion

    #region Unity methods
    void Awake()
    {
        inputProvider = gameObject.GetComponent<InputProvider>();
    }

    void OnEnable()
    {
        Accelerate.OnHold += inputProvider.OnAccelerateButtonPressed;
        TurnLeft.OnHold += inputProvider.OnTurnLeftButtonPressed;
        TurnRight.OnHold += inputProvider.OnTurnRightButtonPressed;

        Dash.OnPress += inputProvider.OnDashButtonPressed;
        Stone.OnPress += inputProvider.OnStoneSkillButtonPressed;
        Boost.OnPress += inputProvider.OnBoostSkillButtonPressed;
        
        BattleManager.Instance.OnBattleChanged += OnBattleChanged;
        SetUpButtonGuide();
    }

    void OnDisable()
    {
        Accelerate.OnHold -= inputProvider.OnAccelerateButtonPressed;
        TurnLeft.OnHold -= inputProvider.OnTurnLeftButtonPressed;
        TurnRight.OnHold -= inputProvider.OnTurnRightButtonPressed;

        Dash.OnPress -= inputProvider.OnDashButtonPressed;
        Stone.OnPress -= inputProvider.OnStoneSkillButtonPressed;
        Boost.OnPress -= inputProvider.OnBoostSkillButtonPressed;

        BattleManager.Instance.OnBattleChanged -= OnBattleChanged;
    }

    void Update()
    {
        foreach (var item in lastInputType)
        {
            if (item.Value != null)
            {
                bool isHolding = (Time.time - item.Value.Item2 < delayHoldSeconds);
                UpdateHoldTypeButtonState(item.Value.Item1, isHolding, item.Key, item.Value.Item3);
            }
        }

        if (Boost.gameObject.activeSelf)
            ResetButtonState(Boost.gameObject);
        if (Stone.gameObject.activeSelf)
            ResetButtonState(Stone.gameObject);
        if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
        {
            UpdateSkillCooldown();
            UpdateDashCooldown();
        }
    }
    #endregion

    #region Button handling methods
    public GameObject GetSelectedSkillButton()
    {
        return Stone.gameObject.activeSelf ? Stone.gameObject : Boost.gameObject;
    }

    void OnBattleChanged(Battle battle)
    {
        if (BattleManager.Instance.CurrentState == BattleState.Battle_End)
            inputProvider.Me().OnPlayerAction -= OnPlayerAction;
        if (BattleManager.Instance.CurrentState == BattleState.Battle_Countdown)
            inputProvider.Me().OnPlayerAction += OnPlayerAction;
    }

    void OnPlayerAction(PlayerSide side, ISumoAction action, bool isPreExecute)
    {
        if (isPreExecute)
        {
            string actionName = action.GetType().Name;
            if (!lastInputType.ContainsKey(actionName))
                lastInputType.Add(actionName, null);
            lastInputType[actionName] = new(GetHoldTypeButtonByAction(action), Time.time, action.InputUsed);
        }
    }

    private GameObject GetHoldTypeButtonByAction(ISumoAction action)
    {
        if ((action is AccelerateAction) || (action is AccelerateTimeAction))
        {
            return Accelerate.gameObject;
        }
        else if ((action is TurnLeftAction) || (action is TurnLeftAngleAction))
        {
            return TurnLeft.gameObject;
        }
        else if ((action is TurnRightAction) || (action is TurnRightAngleAction))
        {
            return TurnRight.gameObject;
        }
        else if (action is DashAction || action is DashTimeAction)
        {
            return Dash.gameObject;
        }
        else if (action is SkillAction)
        {
            return GetSelectedSkillButton();
        }
        return null;
    }

    private void SetUpButtonGuide()
    {
        foreach (var item in InputProvider.KeyboardBindings[inputProvider.PlayerSide])
        {
            GameObject go = GetHoldTypeButtonByAction(item.Value);
            if (go != null)
            {
                TMP_Text text = go.GetComponentInChildren<TMP_Text>();
                text.SetText($"{go.name}\n({item.Key})");
            }
        }
    }

    private void UpdateDashCooldown()
    {
        Battle battle = BattleManager.Instance.Battle;
        bool IsCooldown = inputProvider.PlayerSide == PlayerSide.Left ? battle.LeftPlayer.IsDashOnCooldown : battle.RightPlayer.IsDashOnCooldown;

        Dash.GetComponentInChildren<Button>().interactable = !IsCooldown;
    }

    private void UpdateSkillCooldown()
    {
        GameObject selectedSkillButton = GetSelectedSkillButton();
        if (selectedSkillButton == null) 
            return;

        Battle battle = BattleManager.Instance.Battle;
        SumoController player = inputProvider.PlayerSide == PlayerSide.Left ? battle.LeftPlayer : battle.RightPlayer;
        
        selectedSkillButton.GetComponentInChildren<Button>().interactable = !player.Skill.IsSkillCooldown;
    }

    public void ResetCooldown()
    {
        // Special Skill
        GameObject selectedSpecialSkillObj = Stone.gameObject.activeSelf ? Stone.gameObject : Boost.gameObject;

        // GameObject maybe null when the engine detached
        if (selectedSpecialSkillObj == null) 
            return;

        Battle battle = BattleManager.Instance.Battle;
        SumoController player = inputProvider.PlayerSide == PlayerSide.Left ? battle.LeftPlayer : battle.RightPlayer;
        selectedSpecialSkillObj.GetComponentInChildren<Button>().interactable = true;

        // Dash
        Dash.GetComponentInChildren<Button>().interactable = true;
    }

    void ResetButtonState(GameObject button)
    {
        button.GetComponent<Button>().interactable = true;
    }

    // Prevent multiple input
    void UpdateHoldTypeButtonState(GameObject buttonObject, bool active, string actionName, InputType? inputType)
    {
        if (buttonObject == null)
            return;

        Button button = buttonObject.GetComponent<Button>();

        if (inputType == null)
        {
            button.interactable = true;
            inputProvider.StateKeyboardAction[actionName] = true;
            return;
        }

        bool targetState = !active;

        if (inputType == InputType.Keyboard)
            button.interactable = targetState;
        else if (inputType == InputType.UI)
            inputProvider.StateKeyboardAction[actionName] = targetState;
        else
        {
            inputProvider.StateKeyboardAction[actionName] = targetState;
            button.interactable = targetState;
        }
    }

    public GameObject SetSkillAvailability(SkillType type)
    {
        GameObject activatedObject = type == SkillType.Boost ? Boost.gameObject : Stone.gameObject;
        GameObject deactivatedObject = type == SkillType.Boost ? Stone.gameObject : Boost.gameObject;
        
        if (activatedObject.activeSelf == true && type == SkillType.Stone)
            Stone.transform.position = Boost.gameObject.transform.position;

        activatedObject.SetActive(true);
        deactivatedObject.SetActive(false);

        return activatedObject;
    }
    #endregion
}