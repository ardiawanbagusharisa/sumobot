using SumoCore;
using SumoLog;
using SumoManager;
using UnityEngine;

namespace SumoHelper
{
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

        public EventLogger(SumoController controller, bool forceSave = true, bool isAction = true)
        {
            this.controller = controller;
            ForceSave = forceSave;
            if (isAction)
            {
                action = new();
            }
        }

        public void Call(ISumoAction action)
        {
            if (!IsActive)
            {
                debounceTime = action.Duration;

                IsActive = true;
                startTime = BattleManager.Instance.ElapsedTime;

                this.action.Action = action;
                this.action.Position = controller.RigidBody.position;
                this.action.Rotation = controller.RigidBody.rotation;
                this.action.LinearVelocity = controller.LastLinearVelocity;
                this.action.AngularVelocity = controller.RigidBody.angularVelocity;

                SaveAction(true);
            }
            // else
            // {
            //     var halfArena = BattleManager.Instance.ArenaRadius / 2;
            //     if (Vector2.Distance(controller.RigidBody.position, this.action.Position) > halfArena)
            //     {
            //         SaveAction(false);
            //     }
            //     if (Mathf.Abs(Mathf.DeltaAngle(controller.RigidBody.rotation, this.action.Rotation)) > 90)
            //     {
            //         SaveAction(false);
            //     }
            // }

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
                this.collision.Position = controller.RigidBody.position;
                this.collision.Rotation = controller.RigidBody.rotation;
                this.collision.LinearVelocity = controller.LastLinearVelocity;
                this.collision.AngularVelocity = controller.RigidBody.angularVelocity;
                SaveCollision(true);
            }


            lastCallTime = BattleManager.Instance.ElapsedTime;
        }

        public void Update()
        {
            if (IsActive && debounceTime != 0f && BattleManager.Instance.ElapsedTime - lastCallTime >= debounceTime)
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
            if (!IsActive || !ForceSave)
                return;

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
                action.Rotation = controller.RigidBody.rotation;
                action.Position = controller.RigidBody.position;
                action.LinearVelocity = controller.LastLinearVelocity;
                action.AngularVelocity = controller.RigidBody.angularVelocity;
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
                collision.Rotation = controller.RigidBody.rotation;
                collision.Position = controller.RigidBody.position;
                collision.LinearVelocity = controller.LastLinearVelocity;
                collision.AngularVelocity = controller.RigidBody.angularVelocity;

                float duration = BattleManager.Instance.ElapsedTime - startTime;
                collision.Duration = duration;
            }
            else
            {
                if (collision.IsActor)
                {
                    if (controller.Side == PlayerSide.Left)
                        target = PlayerSide.Right;
                    else
                        target = PlayerSide.Left;
                }
                else
                    target = null;
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

        public void Save(string customCategory, float time)
        {
            BaseLog log = new()
            {
                Rotation = controller.RigidBody.rotation,
                Position = controller.RigidBody.position,
                LinearVelocity = controller.LastLinearVelocity,
                AngularVelocity = controller.RigidBody.angularVelocity
            };

            LogManager.LogPlayerEvents(
                actor: controller.Side,
                startedAt: time,
                updatedAt: time,
                isStart: false,
                category: customCategory,
                data: new()
                {
                    {"Robot", log.ToMap()}
                }
            );
        }
    }
}