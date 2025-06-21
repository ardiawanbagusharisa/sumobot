using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace CoreSumo
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
        #endregion

        #region Physics Stats Properties
        [Header("Physics Stats")]
        public float StopDelay = 0.5f;           
        public float SlowDownRate = 2.0f; 
        public float Torque = 0.4f;
        public float TurnRate = 1f;
        public float BounceResistance = 1f;
        public float LockReductionMultiplier = 0.9f;
        public float CollisionBaseForce = 4f;
        public float BaseLockDurationMultiplier = 0.5f;
        public MinMax LockDuration = new MinMax(0.8f, 2f);
        public MinMax HalfTurnAngle = new MinMax(0f, 180f);
        public MinMax FullTurnAngle = new MinMax(-360, 360);
        #endregion

        #region General Properties
        [Header("General Info")]
        public PlayerSide Side;
        public SpriteRenderer directionIndicator;
        public InputProvider InputProvider;
        public SumoSkill Skill;
        #endregion

        #region Runtime Variables (readonly) Properties
        public bool isInputDisabled = false;
        public Vector2 LastVelocity { get; private set; } = Vector2.zero;
        public float LastAngularVelocity => robotRigidBody.angularVelocity;
        public float LastDashTime = 0;
        public Vector3 StartPosition;
        public Quaternion StartRotation;

        private bool isMoveDisabled = true;
        private bool isSkillDisabled = true;
        private float reservedMoveSpeed;
        private float reservedDashSpeed;
        private float reserverdBounceResistance;
        private Rigidbody2D robotRigidBody;
        private float moveLockTime = 0f;
        private bool isOutOfArena = false;

        // Derived 
        public bool IsDashActive => LastDashTime == 0 ? false : (LastDashTime + DashDuration) >= BattleManager.Instance.ElapsedTime;
        public float DashCooldownTimer => LastDashTime + DashCooldown - BattleManager.Instance.ElapsedTime;
        public float DashCooldownNormalized => 1 - DashCooldownTimer / DashCooldown;
        public bool IsDashOnCooldown => DashCooldownTimer >= 0f;
        private bool IsMovementLocked => moveLockTime > 0f;
        public LogActorType ActorType => Side == PlayerSide.Left ? LogActorType.LeftPlayer : LogActorType.RightPlayer;

        // Events 
        public event Action<PlayerSide> OnPlayerBounce;
        public event Action<PlayerSide> OnPlayerOutOfArena;
        public event Action<PlayerSide, ISumoAction, bool> OnPlayerAction;
        private Coroutine accelerateOverTimeCoroutine;
        private Coroutine turnOverAngleCoroutine;
        #endregion

        #region Unity Methods
        private void Awake()
        {
            Skill = new SumoSkill(this);
            robotRigidBody = GetComponent<Rigidbody2D>();
            reservedMoveSpeed = MoveSpeed;
            reservedDashSpeed = DashSpeed;
            reserverdBounceResistance = BounceResistance;
        }

        void Update()
        {
            ReadInput();

            LastVelocity = robotRigidBody.linearVelocity;
            LogManager.UpdatePlayerActionLog(Side);
        }

        void FixedUpdate()
        {
            HandleStopping();
            if (IsMovementLocked)
                moveLockTime -= Time.deltaTime;
        }

        void OnCollisionEnter2D(Collision2D collision)
        {
            BounceRule(collision);
        }

        void OnTriggerExit2D(Collider2D collision)
        {
            if (!Application.isPlaying) 
                return;

            if (collision.CompareTag("Arena/Floor") && !isOutOfArena)
            {
                OnPlayerOutOfArena?.Invoke(Side);
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

            UpdateDirectionColor();
            SetSkillEnabled(false);
            SetMovementEnabled(false);
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
            {
                directionIndicator.color = new Color(0, 255, 0);
            }
            else
            {
                directionIndicator.color = new Color(255, 0, 0);
            }
        }
        #endregion

        #region Robot Action Methods
        public void StopCoroutineAction()
        {
            if (accelerateOverTimeCoroutine != null)
                StopCoroutine(accelerateOverTimeCoroutine);
            if (turnOverAngleCoroutine != null)
                StopCoroutine(turnOverAngleCoroutine);
        }
        public void Accelerate(AccelerateType type, float time = float.NaN)
        {
            if (isMoveDisabled || IsMovementLocked)
                return;

            LogManager.CallPlayerActionLog(Side, "Accelerate");

            switch (type)
            {
                case AccelerateType.Default:
                    robotRigidBody.linearVelocity = transform.up * (IsDashActive ? DashSpeed : MoveSpeed);
                    break;
                case AccelerateType.Time:
                    accelerateOverTimeCoroutine = StartCoroutine(AccelerateOverTime(time));
                    break;
            }
        }

        public void Dash(DashType type, float time = float.NaN)
        {
            if (isMoveDisabled || IsMovementLocked)
                return;

            switch (type)
            {
                case DashType.Default:
                    if (!IsDashOnCooldown)
                    {
                        LogManager.CallPlayerActionLog(Side, "Dash");

                        LastDashTime = BattleManager.Instance.ElapsedTime;
                        robotRigidBody.linearVelocity = transform.up * DashSpeed;
                    }
                    break;
                case DashType.Time:
                    accelerateOverTimeCoroutine = StartCoroutine(AccelerateOverTime(time, isDash: true));
                    break;
            }
        }

        public void Turn(TurnType type = TurnType.Angle, float angle = float.NaN)
        {
            switch (type)
            {
                case TurnType.Left:
                    LogManager.CallPlayerActionLog(Side, "TurnLeft");
                    robotRigidBody.MoveRotation(robotRigidBody.rotation + RotateSpeed * Time.fixedDeltaTime);
                    break;
                case TurnType.Right:
                    LogManager.CallPlayerActionLog(Side, "TurnRight");
                    robotRigidBody.MoveRotation(robotRigidBody.rotation + -RotateSpeed * Time.fixedDeltaTime);
                    break;
                case TurnType.LeftAngle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(Mathf.Clamp(angle, HalfTurnAngle.min, FullTurnAngle.max) * 1f));
                    break;
                case TurnType.RightAngle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(Mathf.Clamp(angle, HalfTurnAngle.min, HalfTurnAngle.max) * -1f));
                    break;
                case TurnType.Angle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(Mathf.Clamp(angle, FullTurnAngle.min, FullTurnAngle.max)));
                    break;
            }
        }

        public void UseSkill()
        {
            if (!isSkillDisabled) 
                Skill.Activate(Skill.Type);
        }

        IEnumerator TurnOverAngle(float totalAngle)
        {
            if (totalAngle < 0)
                LogManager.CallPlayerActionLog(Side, "TurnLeft", param: totalAngle.ToString());
            else
                LogManager.CallPlayerActionLog(Side, "TurnRight", param: totalAngle.ToString());

            float rotatedAngle = 0f;
            float duration = Mathf.Abs(totalAngle) * TurnRate / LastVelocity.magnitude;
            
            if (duration > TurnRate)
                duration = TurnRate;

            float speed = totalAngle / duration;

            while (Mathf.Abs(rotatedAngle) < Mathf.Abs(totalAngle) && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing && !IsMovementLocked)
            {
                float delta = speed * Time.deltaTime; 

                if (Mathf.Abs(rotatedAngle + delta) > Mathf.Abs(totalAngle))
                    delta = totalAngle - rotatedAngle;

                transform.Rotate(0, 0, delta); 
                rotatedAngle += delta;

                // LogManager.CallPlayerActionLog(Side, totalAngle < 0 ? "TurnLeft" : "TurnRight"); + Side
                if (totalAngle < 0)
                    LogManager.CallPlayerActionLog(Side, "TurnLeft");
                else
                    LogManager.CallPlayerActionLog(Side, "TurnRight");

                yield return null;
            }
        }

        IEnumerator AccelerateOverTime(float time, bool isDash = false)
        {
            float elapsedTime = 0f;
            float speed = isDash ? DashSpeed : MoveSpeed;

            if (isDash)
                LogManager.CallPlayerActionLog(Side, "Dash", param: time.ToString());
            else
                LogManager.CallPlayerActionLog(Side, "Accelerate", param: time.ToString());

            while (elapsedTime < time && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing && !IsMovementLocked)
            {
                robotRigidBody.linearVelocity = transform.up.normalized * speed;
                elapsedTime += Time.deltaTime;

                if (isDash)
                    LogManager.CallPlayerActionLog(Side, "Dash");
                else
                    LogManager.CallPlayerActionLog(Side, "Accelerate");
                
                yield return null;
            }
        }

        public void SetMovementEnabled(bool value)
        {
            isMoveDisabled = !value;
        }
        public void SetSkillEnabled(bool value)
        {
            isSkillDisabled = !value;
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
            {
                lockDuration *= LockReductionMultiplier;
            }
            moveLockTime = Mathf.Max(moveLockTime, lockDuration);
            return lockDuration;
        }
        #endregion

        #region Robot Physics Methods

        void BounceRule(Collision2D collision)
        {
            if (!Application.isPlaying) 
                return;

            SumoController otherRobot = collision.gameObject.GetComponent<SumoController>();
            if (otherRobot == null) 
                return;

            float actorVelocity = LastVelocity.magnitude;
            float enemyVelocity = otherRobot.LastVelocity.magnitude;

            StopCoroutineAction();
            //otherRobot.StopCoroutineAction();

            // Faster robot handles the bounce 
            if (actorVelocity >= enemyVelocity)
            {
                Vector2 collisionNormal = collision.contacts[0].normal;
                float total = actorVelocity + enemyVelocity + 0.01f;

                if (total < 0.1f) 
                    return;

                float actorImpact = CollisionBaseForce * enemyVelocity / total; 
                float targetImpact = CollisionBaseForce * actorVelocity / total;

                if (Skill.Type == ERobotSkillType.Stone && Skill.IsActive)
                    targetImpact = enemyVelocity / total;

                if (otherRobot.Skill.Type == ERobotSkillType.Stone && otherRobot.Skill.IsActive)
                    actorImpact = actorVelocity / total;

                actorImpact *= otherRobot.BounceResistance;
                targetImpact *= BounceResistance;

                Bounce(collisionNormal, actorImpact); 
                otherRobot.Bounce(-collisionNormal, targetImpact); 

                OnPlayerBounce?.Invoke(Side);
                otherRobot.OnPlayerBounce?.Invoke(Side);

                float actorLockDuration = LockMovement(isActor: true, actorImpact);
                float targetLockDuration = otherRobot.LockMovement(isActor: false, targetImpact);

                Debug.Log($"[BounceRule]\nActor=>{Side},Target=>{otherRobot.Side}\nActorVelocity=>{actorVelocity},TargetVelocity=>{enemyVelocity}\nActorCurrentSkill=>{Skill.Type}, TargetCurrentSkill=>{otherRobot.Skill.Type}\nActorImpact=>{actorImpact}, TargetImpact=>{targetImpact}");

                LogManager.LogRoundEvent(
                    actor: Side.ToLogActorType(),
                    target: otherRobot.Side.ToLogActorType(),
                    data: new Dictionary<string, object>()
                    {
                        {"type", "Bounce" },
                        {"actor", new Dictionary<string, object>() {
                            {"impact", actorImpact},
                            {"velocity", new Dictionary<string, object>() {
                                {"x", LastVelocity.x},
                                {"y", LastVelocity.y}}
                            },
                            {"isCurrentSkillActive", Skill.IsActive},
                            {"bounceResistance", BounceResistance},
                            {"lockDuration", actorLockDuration},
                        } },
                        {"target", new Dictionary<string, object>() {
                            {"impact", targetImpact},
                            {"velocity", new Dictionary<string, object>() {
                                {"x", otherRobot.LastVelocity.x},
                                {"y", otherRobot.LastVelocity.y}}
                            },
                            {"isCurrentSkillActive", otherRobot.Skill.IsActive},
                            {"bounceResistance", otherRobot.BounceResistance},
                            {"lockDuration", targetLockDuration},
                        } },
                    });
            }
        }

        public void Bounce(Vector2 direction, float force)
        {
            robotRigidBody.AddForce(force * direction, ForceMode2D.Impulse);
            robotRigidBody.AddTorque(force * Torque, ForceMode2D.Impulse);
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
                OnPlayerAction?.Invoke(Side, action, true);
                action.Execute(this);
                OnPlayerAction?.Invoke(Side, action, false);
            }
        }
        #endregion
    }
}