
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

        foreach (var item in actions)
        {
            if (item is AccelerateAction)
                SetButtonState(Accelerate.gameObject, true);
            if (item is TurnLeftAction)
                SetButtonState(TurnLeft.gameObject, true);
            if (item is TurnRightAction)
                SetButtonState(TurnRight.gameObject, true);
        }
    }

    void SetButtonState(GameObject button, bool active)
    {
        var targetColor = active ? SelectedColor : NormalColor;

        button.GetComponent<Button>().image.color = targetColor;
    }

    // Set active to button about what's skill can be used for player
    public TMP_Text SetSkillAvailability(ERobotSkillType type)
    {
        if (type == ERobotSkillType.Boost)
        {
            Boost.gameObject.SetActive(true);

            Stone.gameObject.SetActive(false);
            return Boost.gameObject.GetComponentInChildren<TMP_Text>();
        }
        else
        {
            Stone.gameObject.SetActive(true);

            // In the debug mode, the Stone is actually in the center of the Button Area,
            // we need to swap Stone position with Boost in order to have a neat position
            Stone.transform.position = Boost.gameObject.transform.position;

            Boost.gameObject.SetActive(false);
            return Stone.gameObject.GetComponentInChildren<TMP_Text>();
        }
    }
}