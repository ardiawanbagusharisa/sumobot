using System;
using CoreSumo;
using UnityEngine;

public class EventLogger
{
    public bool IsActive;
    public bool ForceSave = true;

    private float debounceTime;
    private float lastCallTime;

    private float startTime;
    private CollisionLog collision;
    private readonly ActionLog action;
    private readonly SumoController controller;

    public EventLogger(SumoController controller, float debounceTime, bool forceSave = true)
    {
        this.controller = controller;
        this.debounceTime = debounceTime;
        ForceSave = forceSave;
        action = new();
    }

    public EventLogger(SumoController controller, bool forceSave)
    {
        ForceSave = forceSave;
        this.controller = controller;
    }

    public void Call(ISumoAction action)
    {
        if (!IsActive)
        {
            IsActive = true;
            startTime = BattleManager.Instance.ElapsedTime;

            this.action.Action = action;
            this.action.Position = controller.transform.position;
            this.action.Rotation = controller.transform.rotation.eulerAngles.z;
            this.action.LinearVelocity = controller.LastVelocity;
            this.action.AngularVelocity = controller.LastAngularVelocity;

            SaveAction(true);
        }

        lastCallTime = BattleManager.Instance.ElapsedTime;
    }

    public void Call(CollisionLog collision = null)
    {
        if (this.collision == null && collision != null)
        {
            this.collision = collision;
            debounceTime = collision.LockDuration;
        }

        if (!IsActive)
        {
            IsActive = true;
            startTime = BattleManager.Instance.ElapsedTime;

            collision.Position = controller.transform.position;
            collision.Rotation = controller.transform.rotation.eulerAngles.z;
            collision.LinearVelocity = controller.LastVelocity;
            collision.AngularVelocity = controller.LastAngularVelocity;

            SaveCollision(true);
        }

        lastCallTime = BattleManager.Instance.ElapsedTime;
    }

    public void Update()
    {
        if (IsActive && BattleManager.Instance.ElapsedTime - lastCallTime >= debounceTime)
        {
            IsActive = false;

            if (action != null)
                SaveAction(false);
            else if (collision != null)
                SaveCollision(false);
        }
    }

    public void ForceStopAndSave()
    {
        IsActive = false;
        if (action != null)
            SaveAction(false);
        if (collision != null)
            SaveCollision(false);
    }

    public void SaveAction(bool isStart = false)
    {
        if (!isStart)
        {
            action.Rotation = controller.transform.rotation.eulerAngles.z;
            action.Position = controller.transform.position;
            action.LinearVelocity = controller.LastVelocity;
            action.AngularVelocity = controller.LastAngularVelocity;

            float duration = BattleManager.Instance.ElapsedTime - startTime;
            action.Duration = duration;
        }

        LogManager.LogPlayerEvents(
            actor: controller.Side,
            startedAt: startTime,
            isStart: isStart,
            category: "Action",
            data: action.ToMap()
        );
    }

    public void SaveCollision(bool isStart = false)
    {
        Debug.Log("SaveCollision");

        PlayerSide? target = null;

        if (!isStart)
        {
            collision.Rotation = controller.transform.rotation.eulerAngles.z;
            collision.Position = controller.transform.position;
            collision.LinearVelocity = controller.LastVelocity;
            collision.AngularVelocity = controller.LastAngularVelocity;

            float duration = BattleManager.Instance.ElapsedTime - startTime;
            collision.Duration = duration;
        }
        else
        {
            if (collision.IsActor)
            {
                if (controller.Side == PlayerSide.Left)
                {
                    target = PlayerSide.Right;
                }
                else
                {
                    target = PlayerSide.Left;
                }
            }
            else
            {
                target = null;
            }
        }

        LogManager.LogPlayerEvents(
            actor: controller.Side,
            target: target,
            startedAt: startTime,
            isStart: isStart,
            category: "Collision",
            data: collision.ToMap()
        );
    }
}
