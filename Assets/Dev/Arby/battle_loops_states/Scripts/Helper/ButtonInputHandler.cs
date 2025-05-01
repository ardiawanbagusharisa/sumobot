
using System;
using System.Collections.Generic;
using System.Linq;
using CoreSumoRobot;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class ButtonInputHandler : MonoBehaviour
{

    private List<Button> buttons = new List<Button>();

    private InputProvider inputProvider;

    void Awake()
    {

        inputProvider = gameObject.GetComponent<InputProvider>();


        buttons = GetComponentsInChildren<Button>().Where((s) => s.gameObject != gameObject).ToList();
        InitializeListener();

    }

    private void InitializeListener()
    {
        foreach (var item in buttons)
        {
            switch (item.tag)
            {
                case "Button/Accelerate":
                    item.AddComponent<ButtonPointerHandler>().OnHold += inputProvider.OnAccelerateButtonPressed;
                    break;
                case "Button/TurnLeft":
                    item.AddComponent<ButtonPointerHandler>().OnHold += inputProvider.OnTurnLeft;
                    break;
                case "Button/TurnRight":
                    item.AddComponent<ButtonPointerHandler>().OnHold += inputProvider.OnTurnRight;
                    break;
                case "Button/Dash":
                    item.AddComponent<ButtonPointerHandler>().OnPress += inputProvider.OnDashButtonPressed;
                    break;
                case "Button/Stone":
                    item.AddComponent<ButtonPointerHandler>().OnPress += inputProvider.OnStoneSkill;
                    break;
                case "Button/Boost":
                    item.AddComponent<ButtonPointerHandler>().OnPress += inputProvider.OnBoostSkill;
                    break;
            }
        }
    }

    void OnDestroy()
    {
        foreach (Button item in buttons)
        {
            switch (item.tag)
            {
                case "Button/Accelerate":
                    item.AddComponent<ButtonPointerHandler>().OnHold -= inputProvider.OnAccelerateButtonPressed;
                    break;
                case "Button/TurnLeft":
                    item.AddComponent<ButtonPointerHandler>().OnHold -= inputProvider.OnTurnLeft;
                    break;
                case "Button/TurnRight":
                    item.AddComponent<ButtonPointerHandler>().OnHold -= inputProvider.OnTurnRight;
                    break;
                case "Button/Dash":
                    item.AddComponent<ButtonPointerHandler>().OnPress -= inputProvider.OnDashButtonPressed;
                    break;
                case "Button/Stone":
                    item.AddComponent<ButtonPointerHandler>().OnPress -= inputProvider.OnStoneSkill;
                    break;
                case "Button/Boost":
                    item.AddComponent<ButtonPointerHandler>().OnPress -= inputProvider.OnBoostSkill;
                    break;
            }
        }
    }
}