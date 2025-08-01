using NUnit.Framework;
using SumoCore;
using SumoLog;
using SumoManager;

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
        private readonly SumoController enemyController = null;

        public EventLogger(SumoController controller, SumoController enemyController = null, bool forceSave = true, bool isAction = true)
        {
            this.controller = controller;
            this.enemyController = enemyController;
            ForceSave = forceSave;
            if (isAction)
            {
                action = new();
            }
            else
            {
                Collision = new();
                SetState();
            }
        }

        public static EventLogger CreateCollisionLog(SumoController controller, SumoController enemyController)
        {
            return new(controller, enemyController, forceSave: false, isAction: false);
        }

        public void Call(ISumoAction action, PeriodicState state = PeriodicState.Start)
        {
            if (!IsActive)
            {
                this.action.Action = action;
                debounceTime = action.Duration;
                startTime = BattleManager.Instance.ElapsedTime;
                SaveAction(state);

                IsActive = true;
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
                SaveCollision(PeriodicState.Start);

                IsActive = true;
                startTime = BattleManager.Instance.ElapsedTime;
            }

            lastCallTime = BattleManager.Instance.ElapsedTime;
        }

        public void Update()
        {
            if (Collision == null) return;

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
            SetState();

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
                SetState();

                float duration = BattleManager.Instance.ElapsedTime - startTime;
                Collision.Duration = duration;
            }
            else
            {
                if (Collision.IsActor && !Collision.IsTieBreaker)
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
                Rotation = Normalize360(controller.RigidBody.rotation),
                Position = controller.RigidBody.position,
                LinearVelocity = controller.RigidBody.linearVelocity.magnitude,
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

        public void SetState()
        {
            RobotLog log = Collision != null ? Collision : action;

            log.Robot.Position = controller.RigidBody.position;
            log.Robot.Rotation = Normalize360(controller.RigidBody.rotation);
            log.Robot.LinearVelocity = controller.RigidBody.linearVelocity.magnitude;
            log.Robot.AngularVelocity = controller.RigidBody.angularVelocity;
            log.Robot.IsDashActive = controller.IsDashActive;
            log.Robot.IsSkillActive = controller.Skill.IsActive;
            log.Robot.IsOutFromArena = controller.IsOutOfArena;

            log.EnemyRobot.Position = enemyController.RigidBody.position;
            log.EnemyRobot.Rotation = Normalize360(enemyController.RigidBody.rotation);
            log.EnemyRobot.LinearVelocity = enemyController.RigidBody.linearVelocity.magnitude;
            log.EnemyRobot.AngularVelocity = enemyController.RigidBody.angularVelocity;
            log.EnemyRobot.IsDashActive = enemyController.IsDashActive;
            log.EnemyRobot.IsSkillActive = enemyController.Skill.IsActive;
            log.EnemyRobot.IsOutFromArena = enemyController.IsOutOfArena;
        }

        float Normalize360(float angle)
        {
            angle %= 360f;
            if (angle < 0) angle += 360f;
            return angle;
        }
    }
}