
using System;
using System.Collections.Generic;
using System.Linq;
using CoreSumoRobot;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.EventSystems;
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

    void FixedUpdate()
    {
        var actions = inputProvider.GetInput();

        GameObject selected = null;

        foreach (var item in actions)
        {
            if (item is AccelerateAction)
            {
                selected = Accelerate.gameObject;
            }
            else if (item is TurnLeftAction)
            {
                selected = TurnLeft.gameObject;
            }
            else if (item is TurnRightAction)
            {
                selected = TurnRight.gameObject;
            }

            // Dash, Stone, And Boost are not implemented because it's a press approach,
            // Would better if use cooldown
        }

        // Update the selected object (or null if nothing matched)
        if (EventSystem.current.currentSelectedGameObject != selected)
        {
            EventSystem.current.SetSelectedGameObject(selected);
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