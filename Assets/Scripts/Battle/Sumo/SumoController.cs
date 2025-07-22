using System.Collections;
using System.Collections.Generic;
using System.Linq;
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
        public float StopTreshold = 0.1f;
        public float SlowDownRate = 2.0f;
        public float Torque = 0.2f;
        public float TurnRate = 0.8f;
        public float BounceResistance = 1f;
        public float LockReductionMultiplier = 0.9f;
        public float CollisionBaseForce = 4f;
        public float BaseLockDurationMultiplier = 0.5f;
        public MinMax LockDuration = new(0.8f, 2f);
        public MinMax HalfTurnAngle = new(0f, 180f);
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
        public float LastAngularVelocity => robotRigidBody.angularVelocity;
        public Vector3 StartPosition;
        public Quaternion StartRotation;
        public bool IsSkillDisabled = true;
        private float reservedMoveSpeed;
        private float reservedDashSpeed;
        private float reserverdBounceResistance;
        private Rigidbody2D robotRigidBody;
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
        private Coroutine accelerateOverTimeCoroutine;
        private Coroutine turnOverAngleCoroutine;

        // Actions
        private Queue<ISumoAction> Actions = new();
        #endregion

        #region Unity Methods
        private void Awake()
        {
            robotRigidBody = GetComponent<Rigidbody2D>();
            reservedMoveSpeed = MoveSpeed;
            reservedDashSpeed = DashSpeed;
            reserverdBounceResistance = BounceResistance;
        }

        void Update()
        {
            LogManager.UpdateActionLog(Side);

            InputProvider?.ReadKeyboardInput();

            LastLinearVelocity = robotRigidBody.linearVelocity;

            if (collisionLogger != null && collisionLogger.IsActive)
                collisionLogger.Update();

            if (IsMovementDisabled)
                moveLockTime -= Time.deltaTime;
        }

        void FixedUpdate()
        {
            HandleStopping();
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
            robotRigidBody.linearVelocity = Vector2.zero;
            robotRigidBody.angularVelocity = 0;
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

        public void StopCoroutineAction()
        {
            if (accelerateOverTimeCoroutine != null)
                StopCoroutine(accelerateOverTimeCoroutine);
            if (turnOverAngleCoroutine != null)
                StopCoroutine(turnOverAngleCoroutine);
        }

        public void Accelerate(ISumoAction action)
        {
            if (IsMovementDisabled || IsDashActive)
                return;

            if (accelerateOverTimeCoroutine != null)
            {
                StopCoroutine(accelerateOverTimeCoroutine);
                accelerateOverTimeCoroutine = null;
            }

            Log(action);

            accelerateOverTimeCoroutine = StartCoroutine(AccelerateRoutine(action));
        }

        public void Turn(ISumoAction action)
        {
            if (turnOverAngleCoroutine != null)
            {
                StopCoroutine(turnOverAngleCoroutine);
                turnOverAngleCoroutine = null;
            }

            Log(action);

            turnOverAngleCoroutine = StartCoroutine(TurnRoutine(action));
        }

        public void Dash(ISumoAction action)
        {
            if (IsMovementDisabled || IsDashOnCooldown)
                return;


            Log(action);

            LastDashTime = time;
            if (accelerateOverTimeCoroutine != null)
            {
                StopCoroutine(accelerateOverTimeCoroutine);
                accelerateOverTimeCoroutine = null;
            }
            robotRigidBody.linearVelocity = transform.up * DashSpeed;

            Log(action);
        }

        IEnumerator TurnRoutine(ISumoAction action)
        {

            float angularSpeed;

            switch (action.Type)
            {
                case ActionType.TurnLeft:
                    angularSpeed = Mathf.Abs(RotateSpeed * TurnRate);
                    break;
                case ActionType.TurnRight:
                    angularSpeed = -Mathf.Abs(RotateSpeed * TurnRate);
                    break;
                default:
                    yield break;
            }

            float elapsedTime = 0f;

            while (elapsedTime < action.Duration &&
                   BattleManager.Instance != null &&
                   BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing &&
                   !IsMovementDisabled)
            {
                float deltaTime = Time.deltaTime;
                float rotationThisFrame = angularSpeed * deltaTime;

                float newRotation = robotRigidBody.rotation + rotationThisFrame;
                robotRigidBody.MoveRotation(newRotation);

                elapsedTime += deltaTime;
                yield return null;

                Log(action);
            }
        }

        IEnumerator AccelerateRoutine(ISumoAction action)
        {
            float elapsedTime = 0f;
            float speed = MoveSpeed;

            while (elapsedTime < action.Duration && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing && !IsMovementDisabled)
            {
                robotRigidBody.linearVelocity = transform.up.normalized * speed;
                elapsedTime += Time.deltaTime;

                Log(action);
                yield return null;
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

        public float LockMovement(bool isActor, float force)
        {
            float lockDuration = Mathf.Clamp(force * BaseLockDurationMultiplier, LockDuration.min, LockDuration.max);
            if (isActor)
                lockDuration *= LockReductionMultiplier;
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

            StopCoroutineAction();

            // Faster robot handles the logic of assignment bounce and logging
            if (actorVelocity >= enemyVelocity)
            {
                LogManager.FlushActionLog();

                float actorImpact = Bounce(collisionNormal, enemyVelocity, actorVelocity, enemyRobot);
                float targetImpact = enemyRobot.Bounce(-collisionNormal, actorVelocity, enemyVelocity, this);

                float actorLockDuration = LockMovement(true, actorImpact);
                float targetLockDuration = enemyRobot.LockMovement(false, targetImpact);

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

                EventParameter sideParam = new(sideParam: Side);
                Events[OnBounce]?.Invoke(sideParam);
                enemyRobot.Events[OnBounce]?.Invoke(sideParam);

                Debug.Log($"[BounceRule]\nActor=>{Side},Target=>{enemyRobot.Side}\nActorVelocity=>{actorVelocity},TargetVelocity=>{enemyVelocity}\nActorCurrentSkill=> {Skill.Type} isActive:{Skill.IsActive}, TargetCurrentSkill=>{enemyRobot.Skill.Type} isActive: {enemyRobot.Skill.IsActive} \nActorImpact=>{actorImpact}, TargetImpact=>{targetImpact}");
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
                torque = 0;
            else
                torque = impact;


            robotRigidBody.AddForce(impact * direction, ForceMode2D.Impulse);
            robotRigidBody.AddTorque(torque * Torque, ForceMode2D.Impulse);
            return impact;
        }

        public void FreezeMovement()
        {
            robotRigidBody.constraints = RigidbodyConstraints2D.FreezePosition;
        }

        public void ResetFreezeMovement()
        {
            robotRigidBody.constraints = RigidbodyConstraints2D.None;
        }

        public void ResetBounceResistance()
        {
            BounceResistance = reserverdBounceResistance;
        }

        private void HandleStopping()
        {
            if (robotRigidBody.linearVelocity.magnitude < StopTreshold)
            {
                robotRigidBody.linearVelocity = Vector2.zero;
                return;
            }

            if (Time.time > LastDashTime + StopDelay)
            {
                robotRigidBody.linearVelocity = Vector2.Lerp(robotRigidBody.linearVelocity, Vector2.zero, SlowDownRate * Time.deltaTime);
                robotRigidBody.angularVelocity = Mathf.Lerp(robotRigidBody.angularVelocity, 0, SlowDownRate * Time.deltaTime);
            }
        }
        #endregion

        #region Robot Movement Input

        public void FlushInput()
        {
            if (InputProvider == null || isInputDisabled) return;

            foreach (var action in InputProvider.FlushAction())
                Actions.Enqueue(action);
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