
using CoreSumoRobot;
using UnityEngine;

public class ButtonInputHanlder : MonoBehaviour
{
    public ButtonHoldHandler Accelerate;
    public ButtonHoldHandler TurnLeft;
    public ButtonHoldHandler TurnRight;
    public ButtonHoldHandler Dash;
    public ButtonHoldHandler Stone;
    public ButtonHoldHandler Boost;

    private InputProvider inputProvider;

    void Awake()
    {
        inputProvider = gameObject.GetComponent<InputProvider>();
    }

    void Start()
    {
        Accelerate.OnHold += inputProvider.OnAccelerateButtonPressed;
        TurnLeft.OnHold += inputProvider.OnTurnLeft;
        TurnRight.OnHold += inputProvider.OnTurnRight;
        Dash.OnPress += inputProvider.OnDashButtonPressed;
        Stone.OnPress += inputProvider.OnStoneSkill;
        Boost.OnPress += inputProvider.OnBoostSkill;
    }

    void OnDestroy()
    {
        Accelerate.OnHold -= inputProvider.OnAccelerateButtonPressed;
        TurnLeft.OnHold -= inputProvider.OnTurnLeft;
        TurnRight.OnHold -= inputProvider.OnTurnRight;
        Dash.OnPress -= inputProvider.OnDashButtonPressed;
        Stone.OnPress -= inputProvider.OnStoneSkill;
        Boost.OnPress -= inputProvider.OnBoostSkill;
    }
}