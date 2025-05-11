
using CoreSumoRobot;
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
        InitializeListener();
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

        // Directly change the image color (you can also play with the button's animator if preferred)
        button.GetComponent<Button>().image.color = targetColor;
    }

    public void SetSkillAvailability(ERobotSkillType type)
    {
        if (type == ERobotSkillType.Boost)
        {
            Boost.gameObject.SetActive(true);
            Stone.gameObject.SetActive(false);
        }
        else
        {
            Stone.gameObject.SetActive(true);
            Boost.gameObject.SetActive(false);
        }
    }

    private void InitializeListener()
    {
        Accelerate.OnHold += inputProvider.OnAccelerateButtonPressed;
        TurnLeft.OnHold += inputProvider.OnTurnLeftButtonPressed;
        TurnRight.OnHold += inputProvider.OnTurnRightButtonPressed;

        Dash.OnPress += inputProvider.OnDashButtonPressed;
        Stone.OnPress += inputProvider.OnStoneSkillButtonPressed;
        Boost.OnPress += inputProvider.OnBoostSkillButtonPressed;
    }

    void OnDestroy()
    {
        Accelerate.OnHold -= inputProvider.OnAccelerateButtonPressed;
        TurnLeft.OnHold -= inputProvider.OnTurnLeftButtonPressed;
        TurnRight.OnHold -= inputProvider.OnTurnRightButtonPressed;

        Dash.OnPress -= inputProvider.OnDashButtonPressed;
        Stone.OnPress -= inputProvider.OnStoneSkillButtonPressed;
        Boost.OnPress -= inputProvider.OnBoostSkillButtonPressed;
    }
}