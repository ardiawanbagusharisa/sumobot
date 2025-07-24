using System.Collections.Generic;
using SumoHelper;
using SumoInput;
using SumoLog;
using SumoManager;
using UnityEngine;

namespace SumoCore
{
    public enum PlayerSide
    {
        Left,
        Right,
    }

    public class SumoController : MonoBehaviour
    {
        #region Robot Stats Properties
        [Header("Robot Stats")]
        public float MoveSpeed = 4.0f;
        public float RotateSpeed = 200.0f;
        public float DashSpeed = 5.0f;
        public float DashDuration = 0.5f;
        public float DashCooldown = 1.0f;
        public SumoSkill Skill;
        #endregion

        #region Physics Stats Properties
        [Header("Physics Stats")]
        public float StopDelay = 0.5f;
        public float AngularStopDelay = 0.5f;
        public float StopTreshold = 0.1f;
        public float SlowDownRate = 2.0f;
        public float Torque = 0.2f;
        public float TurnRate = 0.3f;
        public float BounceResistance = 1f;
        public float LockActorMultiplier = 0.95f;
        public float CollisionBaseForce = 4f;
        #endregion

        #region General Properties
        [Header("General Info")]
        public PlayerSide Side;
        public InputProvider InputProvider;
        public SumoCostume Costume;
        #endregion

        #region Runtime (readonly) Properties
        public bool isInputDisabled = false;
        public Vector2 LastLinearVelocity { get; private set; } = Vector2.zero;
        public Vector3 StartPosition;
        public Quaternion StartRotation;
        public bool IsSkillDisabled = true;
        private float reservedMoveSpeed;
        private float reservedDashSpeed;
        private float reserverdBounceResistance;
        public Rigidbody2D RigidBody;
        private float moveLockTime = 0f;
        private bool isOutOfArena = false;
        private EventLogger collisionLogger;
        private float time => BattleManager.Instance.ElapsedTime;
        // Derived 
        public bool IsDashActive => LastDashTime != 0f && (LastDashTime + DashDuration) >= time;
        public float DashCooldownTimer => LastDashTime + DashCooldown - time;
        public float DashCooldownNormalized => 1 - DashCooldownTimer / DashCooldown;
        public bool IsDashOnCooldown => DashCooldownTimer >= 0f;
        public bool IsMovementDisabled => (BattleManager.Instance != null && BattleManager.Instance.CurrentState != BattleState.Battle_Ongoing) || moveLockTime > 0f;
        public float LastDashTime = 0;

        // Events 
        public EventRegistry Events = new();
        public const string OnBounce = "OnBounce"; // [Side]
        public const string OnOutOfArena = "OnOutOfArena"; // [Side]
        public const string OnAction = "OnAction"; // [Side, ISumoAction, bool]
        public const string OnSkillAssigned = "OnSkillAssigned"; // [Side, ISumoAction, bool]
        #endregion


        // Actions
        #region Action runtime properties
        private Queue<ISumoAction> Actions = new();

        // Accelerate
        private Vector2 movementVelocity = Vector2.zero;
        private float accelerateTimeRemaining = 0f;
        private ISumoAction lastAccelerateAction;

        // Turn
        private float remainingAngle;
        private int rotationDirection;
        private bool isTurning = false;

        private ISumoAction lastTurnAction;
        #endregion


        #region Unity Methods
        private void Awake()
        {
            RigidBody = GetComponent<Rigidbody2D>();
            reservedMoveSpeed = MoveSpeed;
            reservedDashSpeed = DashSpeed;
            reserverdBounceResistance = BounceResistance;
        }

        void Update()
        {
            if (BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing)
            {
                InputProvider?.ReadKeyboardInput();
            }

            LogManager.UpdateActionLog(Side);

            LastLinearVelocity = RigidBody.linearVelocity;

            if (collisionLogger != null && collisionLogger.IsActive)
                collisionLogger.Update();

            if (IsMovementDisabled)
                moveLockTime -= Time.deltaTime;
        }

        void FixedUpdate()
        {
            HandleStopping();
            HandlingAccelerating();
            HandleTurning();
        }

        void OnCollisionEnter2D(Collision2D collision)
        {
            BounceRule(collision);
        }

        void OnTriggerExit2D(Collider2D collision)
        {
            if (collision.CompareTag("Arena/Floor") && !isOutOfArena)
            {
                Events[OnOutOfArena]?.Invoke(new EventParameter(sideParam: Side));
                isOutOfArena = true;
            }
        }
        #endregion

        #region Robot State Methods
        public void Initialize(PlayerSide side, Transform startPosition)
        {
            Side = side;
            StartPosition = startPosition.position;
            StartRotation = startPosition.rotation;

            Costume = GetComponent<SumoCostume>();
            Costume.UpdateSideColor();

            AssignSkill();
            SetSkillEnabled(false);
        }

        public void AssignSkill(SkillType type = SkillType.Boost)
        {
            Skill = SumoSkill.CreateSkill(this, type);
            Events[OnSkillAssigned].Invoke(new() { SkillType = type, Side = Side, });
        }

        public void Reset()
        {
            transform.position = StartPosition;
            transform.rotation = StartRotation;
            RigidBody.linearVelocity = Vector2.zero;
            RigidBody.angularVelocity = 0;
            isOutOfArena = false;
            LastLinearVelocity = Vector2.zero;
            LastDashTime = 0;
            Skill.Reset();
        }
        #endregion

        #region Robot Action Methods
        public void Log(ISumoAction action)
        {
            LogManager.CallActionLog(Side, action);
        }

        public void StopOngoingAction()
        {
            accelerateTimeRemaining = 0;
            isTurning = false;
        }

        public void Accelerate(ISumoAction action)
        {
            if (IsMovementDisabled)
                return;

            float speed = 0;

            if (action.Type == ActionType.Accelerate)
            {
                if (IsDashActive)
                    return;

                speed = MoveSpeed;
            }
            else if (action.Type == ActionType.Dash)
            {
                if (IsDashOnCooldown)
                    return;

                action.Duration = DashDuration;
                speed = DashSpeed;
                LastDashTime = time;
            }

            lastAccelerateAction = action;
            Log(action);

            float angle = RigidBody.rotation;
            Vector2 direction = Quaternion.Euler(0, 0, angle) * Vector2.up;

            movementVelocity = direction.normalized * speed;
            accelerateTimeRemaining = action.Duration;
        }

        public void Turn(ISumoAction action)
        {
            if (IsMovementDisabled)
                return;

            lastTurnAction = action;
            Log(action);

            float delta = RotateSpeed * TurnRate * action.Duration;

            remainingAngle = delta;
            rotationDirection = action.Type == ActionType.TurnRight ? -1 : 1; // Turn CW (right)
            isTurning = true;

        }

        void HandlingAccelerating()
        {
            if (accelerateTimeRemaining > 0f && lastAccelerateAction != null)
            {
                RigidBody.linearVelocity = movementVelocity;
                accelerateTimeRemaining -= Time.fixedDeltaTime;
                Log(lastAccelerateAction);
            }
            else
            {
                lastAccelerateAction = null;
            }
        }

        void HandleTurning()
        {
            if (!isTurning || IsMovementDisabled || lastTurnAction == null)
                return;

            float angularStep = RotateSpeed * TurnRate * Time.fixedDeltaTime;

            // Clamp step to remaining angle
            float step = Mathf.Min(angularStep, remainingAngle);
            step *= rotationDirection;

            float newRotation = RigidBody.rotation + step;
            RigidBody.MoveRotation(newRotation);

            remainingAngle -= Mathf.Abs(step);

            Log(lastTurnAction);

            if (remainingAngle <= 0.001f)
            {
                lastTurnAction = null;
                isTurning = false;
            }

        }
        public void SetSkillEnabled(bool value)
        {
            IsSkillDisabled = !value;
        }

        public void SetInputEnabled(bool value)
        {
            isInputDisabled = !value;
        }

        public void ResetMoveSpeed()
        {
            MoveSpeed = reservedMoveSpeed;
        }

        public void ResetDashSpeed()
        {
            DashSpeed = reservedDashSpeed;
        }

        public float LockMovement(bool isActor, float myImpact, float enemyImpact)
        {
            if (Skill.Type == SkillType.Stone && Skill.IsActive)
            {
                return 0;
            }

            float avgImpact = Mathf.Clamp01((myImpact + enemyImpact) / 2);
            float lockDuration = avgImpact;
            if (isActor)
                lockDuration *= LockActorMultiplier;
            else
                lockDuration *= 1f + (1f - LockActorMultiplier);

            moveLockTime = Mathf.Max(moveLockTime, lockDuration);
            return lockDuration;
        }
        #endregion

        #region Robot Physics Methods

        void BounceRule(Collision2D collision)
        {
            if (!collision.gameObject.TryGetComponent<SumoController>(out var enemyRobot))
                return;

            float actorVelocity = LastLinearVelocity.magnitude + float.Epsilon;
            float enemyVelocity = enemyRobot.LastLinearVelocity.magnitude + float.Epsilon;

            Vector2 collisionNormal = collision.contacts[0].normal;

            StopOngoingAction();

            // Faster robot handles the logic of assignment bounce and logging
            if (actorVelocity >= enemyVelocity)
            {
                LogManager.FlushActionLog();

                float actorImpact = Bounce(collisionNormal, enemyVelocity, actorVelocity, enemyRobot);
                float targetImpact = enemyRobot.Bounce(-collisionNormal, actorVelocity, enemyVelocity, this);

                float actorLockDuration = LockMovement(true, actorImpact, targetImpact);
                float targetLockDuration = enemyRobot.LockMovement(false, targetImpact, actorImpact);

                CollisionLog actorLog = new()
                {
                    IsActor = true,
                    Impact = actorImpact,
                    LockDuration = actorLockDuration,
                    IsSkillActive = Skill.IsActive,
                    IsDashActive = IsDashActive,
                };

                CollisionLog targetLog = new()
                {
                    IsActor = false,
                    Impact = targetImpact,
                    LockDuration = targetLockDuration,
                    IsSkillActive = enemyRobot.Skill.IsActive,
                    IsDashActive = enemyRobot.IsDashActive,
                };

                LogCollision(actorLog);
                enemyRobot.LogCollision(targetLog);

                EventParameter sideParam = new(sideParam: Side, floatParam: moveLockTime);
                Events[OnBounce]?.Invoke(sideParam);
                enemyRobot.Events[OnBounce]?.Invoke(sideParam);

                Debug.Log($"[BounceRule]\nActor=>{Side},Target=>{enemyRobot.Side}\nActorVelocity=>{actorVelocity},TargetVelocity=>{enemyVelocity}\nActorCurrentSkill=> {Skill.Type} isActive:{Skill.IsActive}, TargetCurrentSkill=>{enemyRobot.Skill.Type} isActive: {enemyRobot.Skill.IsActive} \nActorImpact=>{actorImpact}, TargetImpact=>{targetImpact}\nActorLock=>{actorLockDuration}, TargetLock=>{targetLockDuration}");
            }
        }

        public void LogCollision(CollisionLog log)
        {
            collisionLogger = new EventLogger(this, forceSave: false, isAction: false);
            collisionLogger.Call(log);
        }

        public float Bounce(Vector2 direction, float enemyVelocity, float myVelocity, SumoController enemy)
        {
            float total = enemyVelocity + myVelocity;
            float impact = CollisionBaseForce * enemyVelocity / total;
            float torque;

            if (enemy.Skill.Type == SkillType.Stone && enemy.Skill.IsActive)
                impact = myVelocity / total;

            impact *= enemy.BounceResistance;

            if (Skill.Type == SkillType.Stone && Skill.IsActive)
            {
                torque = 0;
                impact = 0;
            }
            else
                torque = impact;

            if (torque > 0 && impact > 0)
            {
                RigidBody.AddForce(impact * direction, ForceMode2D.Impulse);
                RigidBody.AddTorque(torque * Torque, ForceMode2D.Impulse);
                RigidBody.angularDamping = 0.5f;
            }
            else
            {
                RigidBody.angularVelocity = 0;
            }
            return impact;
        }

        public void FreezeMovement()
        {
            RigidBody.constraints = RigidbodyConstraints2D.FreezePosition;
        }

        public void ResetFreezeMovement()
        {
            RigidBody.constraints = RigidbodyConstraints2D.None;
        }

        public void ResetBounceResistance()
        {
            BounceResistance = reserverdBounceResistance;
        }

        private void HandleStopping()
        {
            if (RigidBody.linearVelocity.magnitude < StopTreshold)
            {
                RigidBody.linearVelocity = Vector2.zero;
                return;
            }

            if (Time.time > LastDashTime + StopDelay)
            {
                RigidBody.linearVelocity = Vector2.Lerp(RigidBody.linearVelocity, Vector2.zero, SlowDownRate * Time.deltaTime);
                RigidBody.angularVelocity = Mathf.Lerp(RigidBody.angularVelocity, 0, SlowDownRate * Time.deltaTime);
            }

            if (Mathf.Abs(RigidBody.angularVelocity) <= AngularStopDelay)
            {
                RigidBody.angularVelocity = 0;
                RigidBody.angularDamping = 0;
            }

            // if (moveLockTime <= 0f)
            // {
            //     RigidBody.angularDamping = 0;
            // }
        }
        #endregion

        #region Robot Movement Input

        public void FlushInput()
        {
            if (InputProvider == null || isInputDisabled) return;

            foreach (var action in InputProvider.FlushAction())
                Actions.Enqueue(action);
        }

        public void ClearInput()
        {
            if (InputProvider == null || isInputDisabled) return;

            InputProvider.ClearCommands();
            Actions.Clear();
        }

        public void OnUpdate()
        {
            if (InputProvider == null) return;
            if (isInputDisabled) return;

            while (Actions.Count > 0)
            {
                ISumoAction action = Actions.Dequeue();

                EventParameter actionParam = new(sideParam: Side, actionParam: action, boolParam: true);
                Events[OnAction]?.Invoke(actionParam);
                action.Execute(this);
                actionParam.Bool = false;
                Events[OnAction]?.Invoke(actionParam);
            }
        }
        #endregion
    }
}