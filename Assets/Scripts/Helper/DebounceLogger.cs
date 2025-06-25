using System;
using System.Collections.Generic;
using BattleLoop;
using CoreSumo;
using UnityEngine;

public class DebouncedLogger
{
    private float debounceTime;
    private float lastCallTime;
    public bool IsActive;

    private float startRotation;
    private Vector2 startPosition;
    private Vector2 startLinearVelocity;
    private float startAngularVelocity;
    private float startTime;
    private ISumoAction action;
    private readonly SumoController controller;

    public DebouncedLogger(SumoController controller, float debounceTime)
    {
        this.controller = controller;
        this.debounceTime = debounceTime;
    }

    public void Call(ISumoAction action)
    {
        this.action ??= action;

        if (!IsActive)
        {
            IsActive = true;
            startTime = BattleManager.Instance.ElapsedTime;
            startPosition = controller.transform.position;
            startRotation = controller.transform.rotation.eulerAngles.z;
            startLinearVelocity = controller.LastVelocity;
            startAngularVelocity = controller.LastAngularVelocity;
            SaveToLog(true);
        }

        lastCallTime = BattleManager.Instance.ElapsedTime;
    }

    public void Update()
    {
        if (IsActive && BattleManager.Instance.ElapsedTime - lastCallTime >= debounceTime)
        {
            IsActive = false;
            SaveToLog(false);
        }
    }

    public void ForceStopAndSave()
    {
        IsActive = false;
        SaveToLog(false);
    }

    public void SaveToLog(bool isStart = false)
    {
        float duration = BattleManager.Instance.ElapsedTime - startTime;
        float endRotation = controller.transform.rotation.eulerAngles.z;
        Vector3 endPosition = controller.transform.position;
        Vector3 endLinearVelocity = controller.LastVelocity;
        float endAngularVelocity = controller.LastAngularVelocity;

        LogManager.LogPlayerEvents(
            actor: controller.Side,
            startedAt: startTime,
            isStart: isStart,
            data: new Dictionary<string, object>()
            {
                        { "Type", action.Type.ToString() },
                        { "Parameter", action.Param },
                        { "Reason", action.Reason },
                        { "Before", new Dictionary<string,object>()
                            {
                                { "AngularVelocity", startAngularVelocity},
                                { "LinearVelocity", new Dictionary<string,float>()
                                    {
                                        {"X",startLinearVelocity.x},
                                        {"Y",startLinearVelocity.y},
                                    }
                                },
                                { "Position", new Dictionary<string,float>()
                                    {
                                        {"X",startPosition.x},
                                        {"Y",startPosition.y},
                                    }
                                },
                                { "Rotation", new Dictionary<string,float>()
                                    {
                                        {"Z",startRotation},
                                    }
                                },
                            }
                        },
                        { "After", new Dictionary<string,object>()
                            {
                                { "AngularVelocity", endAngularVelocity},
                                { "LinearVelocity", new Dictionary<string,float>()
                                    {
                                        {"X",endLinearVelocity.x},
                                        {"Y",endLinearVelocity.y},
                                    }
                                },
                                { "Position", new Dictionary<string,float>()
                                    {
                                        {"X",endPosition.x},
                                        {"Y",endPosition.y},
                                    }
                                },
                                { "Rotation", new Dictionary<string,float>()
                                    {
                                        {"Z",endRotation},
                                    }
                                },
                            }
                        },
                        { "Duration", duration },

            }
        );
    }
}
