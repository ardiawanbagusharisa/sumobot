using BattleLoop;
using CoreSumoRobot;
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

    public Color SelectedColor = Color.grey;
    public Color NormalColor = Color.white;

    private InputProvider inputProvider;


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
    }

    void OnDisable()
    {
        Accelerate.OnHold -= inputProvider.OnAccelerateButtonPressed;
        TurnLeft.OnHold -= inputProvider.OnTurnLeftButtonPressed;
        TurnRight.OnHold -= inputProvider.OnTurnRightButtonPressed;

        Dash.OnPress -= inputProvider.OnDashButtonPressed;
        Stone.OnPress -= inputProvider.OnStoneSkillButtonPressed;
        Boost.OnPress -= inputProvider.OnBoostSkillButtonPressed;
    }


    void FixedUpdate()
    {
        var actions = inputProvider.GetInput();

        // Default to normal state
        SetButtonState(Accelerate.gameObject, false);
        SetButtonState(TurnLeft.gameObject, false);
        SetButtonState(TurnRight.gameObject, false);
        if (Boost.gameObject.activeSelf)
            SetButtonState(Boost.gameObject, false);
        if (Stone.gameObject.activeSelf)
            SetButtonState(Stone.gameObject, false);


        // Loop and check on Holding-Type button
        foreach (var item in actions)
        {
            if (item is AccelerateAction)
            {
                SetHoldButtonState(Accelerate.gameObject, true, item);
            }
            if (item is TurnLeftAction)
            {
                SetHoldButtonState(TurnLeft.gameObject, true, item);
            }
            if (item is TurnRightAction)
            {
                SetHoldButtonState(TurnRight.gameObject, true, item);
            }

        }

        // Handle interactable and cooldown for Skill and Dash
        if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
        {
            UpdateSkillCooldown();
            UpdateDashCooldown();
        }
    }

    private void UpdateDashCooldown()
    {
        bool IsCooldown;
        if (inputProvider.PlayerSide == PlayerSide.Left)
        {
            IsCooldown = BattleManager.Instance.Battle.LeftPlayer.IsDashCooldown;
        }
        else
        {
            IsCooldown = BattleManager.Instance.Battle.RightPlayer.IsDashCooldown;
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

        SumoRobotController player;
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
            selectedSpecialSkillObj.GetComponentInChildren<Button>().interactable = false;
            selectedSpecialSkillObj.GetComponentInChildren<TMP_Text>().SetText(Mathf.CeilToInt(player.Skill.SkillCooldown()).ToString());
        }
        else
        {
            selectedSpecialSkillObj.GetComponentInChildren<Button>().interactable = true;
            selectedSpecialSkillObj.GetComponentInChildren<TMP_Text>().SetText(player.Skill.Type.ToString());
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

        SumoRobotController player;
        if (inputProvider.PlayerSide == PlayerSide.Left)
        {
            player = BattleManager.Instance.Battle.LeftPlayer;
        }
        else
        {
            player = BattleManager.Instance.Battle.RightPlayer;
        }

        selectedSpecialSkillObj.GetComponentInChildren<Button>().interactable = true;
        selectedSpecialSkillObj.GetComponentInChildren<TMP_Text>().SetText(player.Skill.Type.ToString());

        // Dash
        Dash.GetComponentInChildren<Button>().interactable = true;
    }

    void SetButtonState(GameObject button, bool active)
    {
        var targetColor = active ? SelectedColor : NormalColor;

        button.GetComponent<Button>().image.color = targetColor;

        // Reset Interactable State
        button.GetComponent<Button>().interactable = true;
        inputProvider.StateKeyboardAction["AccelerateAction"] = true;
        inputProvider.StateKeyboardAction["TurnRightAction"] = true;
        inputProvider.StateKeyboardAction["TurnLeftAction"] = true;
    }

    // Prevent multiple input
    void SetHoldButtonState(GameObject button, bool active, ISumoAction action)
    {
        SetButtonState(button, active);

        if (active)
        {
            if (action.InputUsed == InputType.Keyboard)
            {
                button.GetComponent<Button>().interactable = false;
            }
            if (action.InputUsed == InputType.UI)
            {
                string name = action.GetType().Name;
                inputProvider.StateKeyboardAction[name] = false;
            }
        }

    }

    // Set active to button about what's skill can be used for player
    public GameObject SetSkillAvailability(ERobotSkillType type)
    {
        if (type == ERobotSkillType.Boost)
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