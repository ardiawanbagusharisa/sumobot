using System;
using System.Collections.Generic;
using BattleLoop;
using CoreSumoRobot;
using UnityEngine;

public class DebouncedLogger
{
    private float debounceTime;
    private float lastCallTime;
    public bool IsActive;

    private Quaternion startRotation;
    private Vector2 startPosition;
    private Vector2 startLinearVelocity;
    private float startAngularVelocity;
    private float startTime;
    private string Name;
    private SumoRobotController controller;

    public DebouncedLogger(SumoRobotController controller, float debounceTime)
    {
        this.controller = controller;
        this.debounceTime = debounceTime;
    }

    public void Call(string name)
    {
        Name = name;
        if (!IsActive)
        {
            IsActive = true;
            startTime = BattleManager.Instance.ElapsedTime;
            startPosition = controller.transform.position;
            startRotation = controller.transform.rotation;
            startLinearVelocity = controller.LastVelocity;
            startAngularVelocity = controller.LastAngularVelocity;
        }

        lastCallTime = BattleManager.Instance.ElapsedTime;
    }

    public void Update()
    {
        if (IsActive && BattleManager.Instance.ElapsedTime - lastCallTime >= debounceTime)
        {
            SaveToLog();
            IsActive = false;
        }
    }

    public void SaveToLog()
    {
        float duration = BattleManager.Instance.ElapsedTime - startTime;
        Quaternion endRotation = controller.transform.rotation;
        Vector3 endPosition = controller.transform.position;
        Vector3 endLinearVelocity = controller.LastVelocity;
        float endAngularVelocity = controller.LastAngularVelocity;

        LogManager.LogRoundEvent(
            actor: controller.Side.ToLogActorType(),
            data: new Dictionary<string, object>()
            {
                        { "type", Name },
                        { "before", new Dictionary<string,object>()
                            {
                                { "angular_velocity", startAngularVelocity},
                                { "linear_velocity", new Dictionary<string,float>()
                                    {
                                        {"x",startLinearVelocity.x},
                                        {"y",startLinearVelocity.y},
                                    }
                                },
                                { "position", new Dictionary<string,float>()
                                    {
                                        {"x",startPosition.x},
                                        {"y",startPosition.y},
                                    }
                                },
                                { "rotation", new Dictionary<string,float>()
                                    {
                                        {"z",startRotation.z},
                                    }
                                },
                            }
                        },
                        { "after", new Dictionary<string,object>()
                            {
                                { "angular_velocity", endAngularVelocity},
                                { "linear_velocity", new Dictionary<string,float>()
                                    {
                                        {"x",endLinearVelocity.x},
                                        {"y",endLinearVelocity.y},
                                    }
                                },
                                { "position", new Dictionary<string,float>()
                                    {
                                        {"x",endPosition.x},
                                        {"y",endPosition.y},
                                    }
                                },
                                { "rotation", new Dictionary<string,float>()
                                    {
                                        {"z",endRotation.z},
                                    }
                                },
                            }
                        },
                        { "duration", duration }
            }
        );
    }
}
