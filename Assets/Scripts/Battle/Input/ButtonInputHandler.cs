using System;
using System.Collections.Generic;
using CoreSumo;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class ButtonInputHandler : MonoBehaviour
{
    public ButtonPointerHandler Accelerate;
    public ButtonPointerHandler TurnLeft;
    public ButtonPointerHandler TurnRight;
    public ButtonPointerHandler Dash;
    public ButtonPointerHandler Stone;
    public ButtonPointerHandler Boost;
    private InputProvider inputProvider;

    private float delayHoldSeconds = 0.08f;

    private Dictionary<string, Tuple<GameObject, float, InputType?>> lastInputType = new Dictionary<string, Tuple<GameObject, float, InputType?>>();

    void Awake()
    {
        inputProvider = gameObject.GetComponent<InputProvider>();
    }

    void OnEnable()
    {
        if (inputProvider != null)
        {
            if (Accelerate != null) Accelerate.OnHold += inputProvider.OnAccelerateButtonPressed;
            if (TurnLeft != null) TurnLeft.OnHold += inputProvider.OnTurnLeftButtonPressed;
            if (TurnRight != null) TurnRight.OnHold += inputProvider.OnTurnRightButtonPressed;

            if (Dash != null) Dash.OnPress += inputProvider.OnDashButtonPressed;
            if (Stone != null) Stone.OnPress += inputProvider.OnStoneSkillButtonPressed;
            if (Boost != null) Boost.OnPress += inputProvider.OnBoostSkillButtonPressed;
        }

        if (BattleManager.Instance != null)
            BattleManager.Instance.OnBattleChanged += OnBattleChanged;

        SetUpButtonGuide();
    }

    void OnDisable()
    {
        if (inputProvider != null)
        {
            if (Accelerate != null) Accelerate.OnHold -= inputProvider.OnAccelerateButtonPressed;
            if (TurnLeft != null) TurnLeft.OnHold -= inputProvider.OnTurnLeftButtonPressed;
            if (TurnRight != null) TurnRight.OnHold -= inputProvider.OnTurnRightButtonPressed;

            if (Dash != null) Dash.OnPress -= inputProvider.OnDashButtonPressed;
            if (Stone != null) Stone.OnPress -= inputProvider.OnStoneSkillButtonPressed;
            if (Boost != null) Boost.OnPress -= inputProvider.OnBoostSkillButtonPressed;

            var me = inputProvider.Me();
            if (me != null)
                me.OnPlayerAction -= OnPlayerAction;
        }

        if (BattleManager.Instance != null)
            BattleManager.Instance.OnBattleChanged -= OnBattleChanged;
    }

    public GameObject GetSelectedSkillButton()
    {
        GameObject selectedSpecialSkillObj;
        if (Stone.gameObject.activeSelf)
        {
            selectedSpecialSkillObj = Stone.gameObject;
        }
        else
        {
            selectedSpecialSkillObj = Boost.gameObject;
        }

        return selectedSpecialSkillObj;
    }


    void OnBattleChanged(Battle battle)
    {
        if (BattleManager.Instance.CurrentState == BattleState.Battle_End)
        {
            inputProvider.Me().OnPlayerAction -= OnPlayerAction;
        }
        if (BattleManager.Instance.CurrentState == BattleState.Battle_Countdown)
        {
            inputProvider.Me().OnPlayerAction += OnPlayerAction;
        }
    }

    void OnPlayerAction(PlayerSide side, ISumoAction action, bool isPreExecute)
    {
        if (isPreExecute)
        {
            var actionName = action.GetType().Name;
            if (!lastInputType.ContainsKey(actionName))
            {
                lastInputType.Add(actionName, null);
            }
            lastInputType[actionName] = new(GetHoldTypeButtonByAction(action), Time.time, action.InputUsed);
        }
    }


    void Update()
    {
        // Handle button interactable state for Hold Type Button
        foreach (var item in lastInputType)
        {
            if (item.Value != null)
            {
                if (Time.time - item.Value.Item2 >= delayHoldSeconds)
                {
                    UpdateHoldTypeButtonState(item.Value.Item1, false, item.Key, item.Value.Item3);
                }
                else
                    UpdateHoldTypeButtonState(item.Value.Item1, true, item.Key, item.Value.Item3);
            }
        }

        // Handle button interactable state for [Skill] and [Dash]
        // Default to normal state
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
            if (GetSelectedSkillButton() != null)
            {
                return GetSelectedSkillButton();
            }
        }
        return null;
    }

    private void SetUpButtonGuide()
    {
        foreach (var item in InputProvider.KeyboardBindings[inputProvider.PlayerSide])
        {
            var go = GetHoldTypeButtonByAction(item.Value);
            if (go != null)
            {
                var text = go.GetComponentInChildren<TMP_Text>();
                text.SetText($"{go.name}\n({item.Key})");
            }
        }
    }

    private void UpdateDashCooldown()
    {
        bool IsCooldown;
        if (inputProvider.PlayerSide == PlayerSide.Left)
        {
            IsCooldown = BattleManager.Instance.Battle.LeftPlayer.IsDashOnCooldown;
        }
        else
        {
            IsCooldown = BattleManager.Instance.Battle.RightPlayer.IsDashOnCooldown;
        }

        if (IsCooldown)
        {
            Dash.GetComponentInChildren<Button>().interactable = false;
        }
        else
        {
            Dash.GetComponentInChildren<Button>().interactable = true;
        }
    }

    private void UpdateSkillCooldown()
    {
        if (GetSelectedSkillButton() == null) return;

        SumoController player;
        if (inputProvider.PlayerSide == PlayerSide.Left)
        {
            player = BattleManager.Instance.Battle.LeftPlayer;
        }
        else
        {
            player = BattleManager.Instance.Battle.RightPlayer;
        }

        if (player.Skill.IsSkillCooldown)
        {
            GetSelectedSkillButton().GetComponentInChildren<Button>().interactable = false;
        }
        else
        {
            GetSelectedSkillButton().GetComponentInChildren<Button>().interactable = true;
        }
    }

    public void ResetCooldown()
    {
        // Special Skill
        GameObject selectedSpecialSkillObj;
        if (Stone.gameObject.activeSelf)
        {
            selectedSpecialSkillObj = Stone.gameObject;
        }
        else
        {
            selectedSpecialSkillObj = Boost.gameObject;
        }

        // GameObject maybe null when the engine detached
        if (selectedSpecialSkillObj == null) return;

        SumoController player;
        if (inputProvider.PlayerSide == PlayerSide.Left)
        {
            player = BattleManager.Instance.Battle.LeftPlayer;
        }
        else
        {
            player = BattleManager.Instance.Battle.RightPlayer;
        }

        selectedSpecialSkillObj.GetComponentInChildren<Button>().interactable = true;

        // Dash
        Dash.GetComponentInChildren<Button>().interactable = true;
    }

    void ResetButtonState(GameObject button)
    {
        button.GetComponent<Button>().interactable = true;
    }

    // Prevent multiple input
    void UpdateHoldTypeButtonState(GameObject button, bool active, string actionName, InputType? inputType)
    {
        if (button == null) return;
        if (inputType == null)
        {
            button.GetComponent<Button>().interactable = true;
            inputProvider.StateKeyboardAction[actionName] = true;
            return;
        }

        if (active)
        {
            if (inputType == InputType.Keyboard)
            {
                button.GetComponent<Button>().interactable = false;
            }
            else if (inputType == InputType.UI)
            {
                inputProvider.StateKeyboardAction[actionName] = false;
            }
            else
            {
                inputProvider.StateKeyboardAction[actionName] = false;
                button.GetComponent<Button>().interactable = false;
            }
        }
        else
        {
            if (inputType == InputType.Keyboard)
            {
                button.GetComponent<Button>().interactable = true;
            }
            else if (inputType == InputType.UI)
            {
                inputProvider.StateKeyboardAction[actionName] = true;
            }
            else
            {
                inputProvider.StateKeyboardAction[actionName] = true;
                button.GetComponent<Button>().interactable = true;
            }
        }

    }

    // Set active to button about what's skill can be used for player
    public GameObject SetSkillAvailability(SkillType type)
    {
        if (type == SkillType.Boost)
        {
            Boost.gameObject.SetActive(true);

            Stone.gameObject.SetActive(false);
            return Boost.gameObject;
        }
        else
        {
            Stone.gameObject.SetActive(true);

            // In the debug mode, the Stone is actually in the center of the Button Area,
            // we need to swap Stone position with Boost in order to have a neat position
            Stone.transform.position = Boost.gameObject.transform.position;

            Boost.gameObject.SetActive(false);
            return Stone.gameObject;
        }
    }
}