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
        public CollisionLog Collision;
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

        public void Call(ISumoAction action, PeriodicState state = PeriodicState.Start)
        {
            if (!IsActive)
            {
                debounceTime = action.Duration;

                IsActive = true;
                startTime = BattleManager.Instance.ElapsedTime;

                this.action.Action = action;
                this.action.Position = controller.RigidBody.position;
                this.action.Rotation = controller.RigidBody.rotation;
                this.action.LinearVelocity = controller.RigidBody.linearVelocity;
                this.action.AngularVelocity = controller.RigidBody.angularVelocity;

                SaveAction(state);
            }

            lastCallTime = BattleManager.Instance.ElapsedTime;
        }

        public void Call()
        {
            if (Collision != null)
            {
                debounceTime = Collision.LockDuration;
            }

            if (!IsActive)
            {
                IsActive = true;
                startTime = BattleManager.Instance.ElapsedTime;
                SaveCollision(PeriodicState.Start);
            }

            lastCallTime = BattleManager.Instance.ElapsedTime;
        }

        public void Update()
        {
            if (IsActive && debounceTime != 0f && lastCallTime != 0f && BattleManager.Instance.ElapsedTime - lastCallTime >= debounceTime)
            {
                IsActive = false;

                if (action != null)
                    SaveAction(PeriodicState.End);
                else if (Collision != null)
                    SaveCollision(PeriodicState.End);
            }
        }

        public void ForceStopAndSave()
        {
            if (!IsActive || !ForceSave)
                return;

            IsActive = false;

            if (action != null)
                SaveAction(PeriodicState.End);
            if (Collision != null)
                SaveCollision(PeriodicState.End);
        }

        public void Kill()
        {
            IsActive = false;
            debounceTime = 0;
            startTime = 0;
            lastCallTime = 0;
        }

        public void SaveAction(PeriodicState state = PeriodicState.Start)
        {
            if (state == PeriodicState.End)
            {
                action.Rotation = controller.RigidBody.rotation;
                action.Position = controller.RigidBody.position;
                action.LinearVelocity = controller.RigidBody.linearVelocity;
                action.AngularVelocity = controller.RigidBody.angularVelocity;
            }

            LogManager.LogPlayerEvents(
                actor: controller.Side,
                startedAt: startTime,
                state: state,
                category: "Action",
                data: action.ToMap()
            );
        }

        public void SaveCollision(PeriodicState state = PeriodicState.Start)
        {
            PlayerSide? target = null;

            if (state == PeriodicState.End)
            {
                Collision.Rotation = controller.RigidBody.rotation;
                Collision.Position = controller.RigidBody.position;
                Collision.LinearVelocity = controller.RigidBody.linearVelocity;
                Collision.AngularVelocity = controller.RigidBody.angularVelocity;

                float duration = BattleManager.Instance.ElapsedTime - startTime;
                Collision.Duration = duration;
            }
            else
            {
                if (Collision.IsActor)
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
                state: state,
                category: "Collision",
                data: Collision.ToMap()
            );
        }

        public void Save(string customCategory, float time)
        {
            BaseLog log = new()
            {
                Rotation = controller.RigidBody.rotation,
                Position = controller.RigidBody.position,
                LinearVelocity = controller.RigidBody.linearVelocity,
                AngularVelocity = controller.RigidBody.angularVelocity
            };

            LogManager.LogPlayerEvents(
                actor: controller.Side,
                startedAt: time,
                updatedAt: time,
                state: PeriodicState.Start,
                category: customCategory,
                data: new()
                {
                    {"Robot", log.ToMap()}
                }
            );
        }
    }
}