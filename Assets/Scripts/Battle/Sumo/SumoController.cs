using System.Collections;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
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
        public float StopTreshold = 0.05f;
        public float SlowDownRate = 2.0f;
        public float Torque = 0.4f;
        public float TurnRate = 1f;
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
        public SpriteRenderer directionIndicator;
        public InputProvider InputProvider;
        #endregion

        #region Runtime (readonly) Properties
        public Bot Bot;
        public bool isInputDisabled = false;
        public Vector2 LastVelocity { get; private set; } = Vector2.zero;
        public float LastAngularVelocity => robotRigidBody.angularVelocity;
        public float LastDashTime = 0;
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

        // Derived 
        public bool IsDashActive => LastDashTime != 0 && (LastDashTime + DashDuration) >= (BattleManager.Instance?.ElapsedTime ?? Time.time);
        public float DashCooldownTimer => LastDashTime + DashCooldown - BattleManager.Instance?.ElapsedTime ?? Time.time;
        public float DashCooldownNormalized => 1 - DashCooldownTimer / DashCooldown;
        public bool IsDashOnCooldown => DashCooldownTimer >= 0f;
        public bool IsMovementDisabled => (BattleManager.Instance != null && BattleManager.Instance.CurrentState != BattleState.Battle_Ongoing) || moveLockTime > 0f;

        // Events 
        public ActionRegistry Actions = new();
        public const string OnPlayerBounce = "OnPlayerBounce"; // [Side]
        public const string OnPlayerOutOfArena = "OnPlayerOutOfArena"; // [Side]
        public const string OnPlayerAction = "OnPlayerAction"; // [Side, ISumoAction, bool]
        private Coroutine accelerateOverTimeCoroutine;
        private Coroutine turnOverAngleCoroutine;
        #endregion

        #region Unity Methods
        private void Awake()
        {
            Bot = GetComponents<Bot>()?.FirstOrDefault((x) => x.enabled);
            robotRigidBody = GetComponent<Rigidbody2D>();
            reservedMoveSpeed = MoveSpeed;
            reservedDashSpeed = DashSpeed;
            reserverdBounceResistance = BounceResistance;

        }

        void Update()
        {
            ReadInput();

            LastVelocity = robotRigidBody.linearVelocity;

            if (collisionLogger != null && collisionLogger.IsActive)
                collisionLogger.Update();

            LogManager.UpdateActionLog(Side);
        }

        void FixedUpdate()
        {
            HandleStopping();
            if (IsMovementDisabled)
                moveLockTime -= Time.deltaTime;
        }

        void OnCollisionEnter2D(Collision2D collision)
        {
            BounceRule(collision);
        }

        void OnTriggerExit2D(Collider2D collision)
        {
            if (collision.CompareTag("Arena/Floor") && !isOutOfArena)
            {
                Actions[OnPlayerOutOfArena]?.Invoke(Side);
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

            AssignSkill();
            UpdateDirectionColor();
            SetSkillEnabled(false);
        }

        public void AssignSkill(SkillType type = SkillType.Boost)
        {
            Skill = SumoSkill.CreateSkill(this, type);
        }

        public void Reset()
        {
            transform.position = StartPosition;
            transform.rotation = StartRotation;
            robotRigidBody.linearVelocity = Vector2.zero;
            robotRigidBody.angularVelocity = 0;
            isOutOfArena = false;
            LastDashTime = 0;
            LastVelocity = Vector2.zero;
            Skill.Reset();
        }

        public void UpdateDirectionColor()
        {
            if (Side == PlayerSide.Left)
                directionIndicator.color = new Color(0, 255, 0);
            else
                directionIndicator.color = new Color(255, 0, 0);
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
            if (IsMovementDisabled || IsDashActive) return;

            Log(action);
            switch (action.Type)
            {
                case ActionType.Accelerate:
                    robotRigidBody.linearVelocity = transform.up * (IsDashActive ? DashSpeed : MoveSpeed);
                    Log(action);
                    break;
                case ActionType.AccelerateWithTime:
                    accelerateOverTimeCoroutine = StartCoroutine(AccelerateOverTime(action));
                    break;
            }
        }

        public void Dash(ISumoAction action)
        {
            if (IsMovementDisabled || IsDashOnCooldown) return;

            Log(action);
            LastDashTime = BattleManager.Instance?.ElapsedTime ?? Time.time;
            robotRigidBody.linearVelocity = transform.up * DashSpeed;
            Log(action);
        }

        public void Turn(ISumoAction action)
        {
            Log(action);

            switch (action.Type)
            {
                case ActionType.TurnLeft:
                    robotRigidBody.MoveRotation(robotRigidBody.rotation + RotateSpeed * Time.fixedDeltaTime);
                    break;
                case ActionType.TurnRight:
                    robotRigidBody.MoveRotation(robotRigidBody.rotation + -RotateSpeed * Time.fixedDeltaTime);
                    break;
                case ActionType.TurnLeftWithAngle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(action));
                    break;
                case ActionType.TurnRightWithAngle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(action));
                    break;
            }
        }

        IEnumerator TurnOverAngle(ISumoAction action)
        {
            float totalAngle;

            switch (action.Type)
            {
                case ActionType.TurnLeftWithAngle:
                    totalAngle = Mathf.Abs((float)action.Param);
                    break;
                case ActionType.TurnRightWithAngle:
                    totalAngle = -Mathf.Abs((float)action.Param);
                    break;
                default:
                    yield break;
            }

            float rotatedAngle = 0f;
            float turnSpeed = TurnRate * 100;

            while (Mathf.Abs(rotatedAngle) < Mathf.Abs(totalAngle) &&
                   BattleManager.Instance != null && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing &&
                   !IsMovementDisabled)
            {
                float delta = turnSpeed * Time.deltaTime;

                if (Mathf.Abs(rotatedAngle + delta * Mathf.Sign(totalAngle)) > Mathf.Abs(totalAngle))
                    delta = Mathf.Abs(totalAngle - rotatedAngle);

                float step = delta * Mathf.Sign(totalAngle);
                transform.Rotate(0, 0, step);
                rotatedAngle += step;

                Log(action);
                yield return null;
            }
        }

        IEnumerator AccelerateOverTime(ISumoAction action, bool isDash = false)
        {
            float elapsedTime = 0f;
            float speed = isDash ? DashSpeed : MoveSpeed;

            Log(action);

            while (elapsedTime < (float)action.Param && BattleManager.Instance != null && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing && !IsMovementDisabled)
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

            float actorVelocity = LastVelocity.magnitude + float.Epsilon;
            float enemyVelocity = enemyRobot.LastVelocity.magnitude + float.Epsilon;

            if ((actorVelocity + enemyVelocity) < 0.01f)
                return;

            Vector2 collisionNormal = collision.contacts[0].normal;

            StopCoroutineAction();

            // Faster robot handles the logic of assignment bounce and logging
            if (actorVelocity >= enemyVelocity)
            {
                float actorImpact = Bounce(collisionNormal, enemyVelocity, actorVelocity, enemyRobot);
                float targetImpact = enemyRobot.Bounce(-collisionNormal, actorVelocity, enemyVelocity, this);

                Actions[OnPlayerBounce]?.Invoke(Side);
                enemyRobot.Actions[OnPlayerBounce]?.Invoke(Side);

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

                Debug.Log($"[BounceRule]\nActor=>{Side},Target=>{enemyRobot.Side}\nActorVelocity=>{actorVelocity},TargetVelocity=>{enemyVelocity}\nActorCurrentSkill=> {Skill.Type} isActive:{Skill.IsActive}, TargetCurrentSkill=>{enemyRobot.Skill.Type} isActive: {enemyRobot.Skill.IsActive} \nActorImpact=>{actorImpact}, TargetImpact=>{targetImpact}");
            }
        }

        public void LogCollision(CollisionLog log)
        {
            collisionLogger = new EventLogger(this, false);
            collisionLogger.Call(log);
        }

        public float Bounce(Vector2 direction, float enemyVelocity, float myVelocity, SumoController enemy)
        {
            float total = enemyVelocity + myVelocity;

            float impact = CollisionBaseForce * enemyVelocity / total;

            if (enemy.Skill.Type == SkillType.Stone && enemy.Skill.IsActive)
                impact = myVelocity / total;

            impact *= enemy.BounceResistance;

            robotRigidBody.AddForce(impact * direction, ForceMode2D.Impulse);
            robotRigidBody.AddTorque(impact * Torque, ForceMode2D.Impulse);
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

        void ReadInput()
        {
            if (InputProvider == null) return;
            if (isInputDisabled) return;

            List<ISumoAction> actions = InputProvider.GetInput();
            foreach (ISumoAction action in actions)
            {
                Actions[OnPlayerAction]?.Invoke(new object[] { Side, action, true });
                action.Execute(this);
                Actions[OnPlayerAction]?.Invoke(new object[] { Side, action, false });
            }
        }
        #endregion
    }
}