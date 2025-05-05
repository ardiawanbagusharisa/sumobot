
using System;
using System.Collections.Generic;
using System.Linq;
using CoreSumoRobot;
using Unity.VisualScripting;
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

    void Awake()
    {

        inputProvider = gameObject.GetComponent<InputProvider>();
        InitializeListener();
    }

    private void InitializeListener()
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