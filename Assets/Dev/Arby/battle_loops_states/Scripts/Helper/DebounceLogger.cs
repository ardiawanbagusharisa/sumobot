using System;
using System.Collections.Generic;
using BattleLoop;
using CoreSumoRobot;
using UnityEngine;

public class DebouncedLogger
{
    private float debounceTime;
    private Action onDebounced;
    private float lastCallTime;
    private bool isActive;

    private Quaternion startRotation;
    private Vector2 startPosition;
    private Vector2 startLinearVelocity;
    private float startTime;
    private string Name;
    private SumoRobotController controller;

    public DebouncedLogger(SumoRobotController controller, float debounceTime)
    {
        this.controller = controller;
        this.debounceTime = debounceTime;
        onDebounced = () =>
        {
            float duration = BattleManager.Instance.ElapsedTime - startTime;
            Quaternion endRotation = controller.transform.rotation;
            Vector3 endPosition = controller.transform.position;
            Vector3 endVelocity = controller.LastVelocity;

            LogManager.LogRoundEvent(
                actor: controller.Side.ToLogActorType(),
                data: new Dictionary<string, object>()
                {
                        { "type", Name },
                        { "before", new Dictionary<string,object>()
                            {
                                { "velocity", new Dictionary<string,float>()
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
                                { "velocity", new Dictionary<string,float>()
                                    {
                                        {"x",endVelocity.x},
                                        {"y",endVelocity.y},
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
        };
    }

    public void Call(string name)
    {
        Name = name;
        if (!isActive)
        {
            isActive = true;
            startTime = BattleManager.Instance.ElapsedTime;
            startPosition = controller.transform.position;
            startRotation = controller.transform.rotation;
            startLinearVelocity = controller.LastVelocity;
        }

        lastCallTime = BattleManager.Instance.ElapsedTime;
    }

    public void Update()
    {
        if (isActive && BattleManager.Instance.ElapsedTime - lastCallTime >= debounceTime)
        {
            onDebounced.Invoke();
            isActive = false;
        }
    }

    // When the battle is ended, some actions are maybe still active (e.g skill) from previous [debounceTime]
    public void AddIncompleteAction()
    {
        if (isActive)
            onDebounced.Invoke();
    }
}
