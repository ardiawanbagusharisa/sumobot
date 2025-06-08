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
        #region Basic Stats
        public int IdInt => Side == PlayerSide.Left ? 0 : 1;
        public float MoveSpeed = 4.0f;
        public float RotateSpeed = 200.0f;
        #endregion

        #region Dash Stats
        public float DashSpeed = 5.0f;
        public float DashDuration = 0.5f;       // Dash duration. 
        public float DashCooldown = 1.0f;       // Dash cooldown.
        #endregion

        #region Physics Stats
        public float StopDelay = 0.5f;           // Time before robot stops.
        public float SlowDownRate = 2.0f;        // Robot's slowdown rate (velocity and rotation decay). 
        public float Torque = 0.4f;
        public float BounceResistance = 1f;
        public float LockReductionMultiplier = 0.9f;
        public float CollisionBaseForce = 4f;
        public float TurnRate = 1f;
        public float BaseLockDurationMultiplier = 0.5f;
        public MinMax LockDuration = new MinMax(0.8f, 2f);
        public MinMax HalfTurnAngle = new MinMax(0f, 180f);
        public MinMax FullTurnAngle = new MinMax(-360, 360);

        public PlayerSide Side;
        #endregion

        public Vector2 LastVelocity { get; private set; } = Vector2.zero;
        public float LastAngularVelocity => robotRigidBody.angularVelocity;
        public bool isInputDisabled = false;
        public SpriteRenderer face;
        public Vector3 StartPosition;
        public Quaternion StartRotation;
        public SumoSkill Skill;
        public InputProvider InputProvider;
        public float LastDashTime = 0;
        public bool IsDashActive => LastDashTime == 0 ? false : (LastDashTime + DashDuration) >= BattleManager.Instance.ElapsedTime;
        public float DashCooldownAmount => LastDashTime + DashCooldown - BattleManager.Instance.ElapsedTime;
        public float DashCooldownAmountNormalized => 1 - DashCooldownAmount / DashCooldown;
        public bool IsDashCooldown => DashCooldownAmount >= 0f;
        public LogActorType ActorType => Side == PlayerSide.Left ? LogActorType.LeftPlayer : LogActorType.RightPlayer;

        //[PlayerSide] is the Actor or BounceMaker
        public event Action<PlayerSide> OnPlayerBounce;

        //[PlayerSide] is the one who get outside from Arena
        public event Action<PlayerSide> OnPlayerOutOfArena;

        //[PlayerSide] is the invoker
        //[ISumoAction] is action that invoked, 
        //[bool] defines true -> preExecute, and false -> postExecute.
        public event Action<PlayerSide, ISumoAction, bool> OnPlayerAction;

        private bool isMoveDisabled = true;
        private bool isSkillDisabled = true;
        private float reservedMoveSpeed;
        private float reservedDashSpeed;
        private float reserverdBounceResistance;
        private Rigidbody2D robotRigidBody;
        private float moveLockTime = 0f;
        private bool IsMovementLocked => moveLockTime > 0f;

        private bool hasOnOutOfArenaInvoked = false;

        private Coroutine accelerateOverTimeCoroutine;
        private Coroutine turnOverAngleCoroutine;

        #region Unity
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
            if (!Application.isPlaying) return;

            if (collision.tag == "Arena/Floor" && !hasOnOutOfArenaInvoked)
            {
                OnPlayerOutOfArena?.Invoke(Side);
                hasOnOutOfArenaInvoked = true;
            }
        }
        #endregion

        #region Robot State
        public void InitializeForBattle(PlayerSide side, Transform startPosition)
        {
            Side = side;
            StartPosition = startPosition.position;
            StartRotation = startPosition.rotation;

            UpdateFaceColor();
            SetSkillEnabled(false);
            SetMovementEnabled(false);
        }

        public void ResetForNewBattle()
        {
            transform.position = StartPosition;
            transform.rotation = StartRotation;
            robotRigidBody.linearVelocity = Vector2.zero;
            robotRigidBody.angularVelocity = 0;
            hasOnOutOfArenaInvoked = false;
            LastDashTime = 0;
            LastVelocity = Vector2.zero;
            Skill.Reset();
        }
        #endregion


        #region Robot Action
        public void StopCoroutineAction()
        {
            if (accelerateOverTimeCoroutine != null)
                StopCoroutine(accelerateOverTimeCoroutine);
            if (turnOverAngleCoroutine != null)
                StopCoroutine(turnOverAngleCoroutine);
        }
        public void Accelerate(AccelerateActionType type, float time = float.NaN)
        {
            if (isMoveDisabled || IsMovementLocked)
            {
                //Debug.Log("Move is disabled.");
                return;
            }

            // Log with debounce strategy
            LogManager.CallPlayerActionLog(Side, "Accelerate");

            switch (type)
            {
                case AccelerateActionType.Default:
                    robotRigidBody.linearVelocity = transform.up * (IsDashActive ? DashSpeed : MoveSpeed);
                    break;
                case AccelerateActionType.Time:
                    accelerateOverTimeCoroutine = StartCoroutine(AccelerateOverTime(time));
                    break;
            }
        }

        public void Dash(DashActionType type, float time = float.NaN)
        {
            if (isMoveDisabled || IsMovementLocked)
            {
                // Debug.Log("Move is disabled.");
                return;
            }

            switch (type)
            {
                case DashActionType.Default:
                    if (!IsDashCooldown)
                    {
                        // Log
                        LogManager.CallPlayerActionLog(Side, "Dash");

                        LastDashTime = BattleManager.Instance.ElapsedTime;
                        robotRigidBody.linearVelocity = transform.up * DashSpeed;
                    }
                    else
                    {
                        // Debug.Log("Dash is on cooldown.");
                    }
                    break;
                case DashActionType.Time:
                    accelerateOverTimeCoroutine = StartCoroutine(AccelerateOverTime(time, isDash: true));
                    break;
            }

        }

        public void Turn(TurnActionType type = TurnActionType.Angle, float angle = float.NaN)
        {
            switch (type)
            {
                case TurnActionType.Left:
                    LogManager.CallPlayerActionLog(Side, "TurnLeft");
                    robotRigidBody.MoveRotation(robotRigidBody.rotation + RotateSpeed * Time.fixedDeltaTime);
                    break;
                case TurnActionType.Right:
                    LogManager.CallPlayerActionLog(Side, "TurnRight");
                    robotRigidBody.MoveRotation(robotRigidBody.rotation + -RotateSpeed * Time.fixedDeltaTime);
                    break;
                case TurnActionType.LeftAngle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(Mathf.Clamp(angle, HalfTurnAngle.min, FullTurnAngle.max) * 1f));
                    break;
                case TurnActionType.RightAngle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(Mathf.Clamp(angle, HalfTurnAngle.min, HalfTurnAngle.max) * -1f));
                    break;
                case TurnActionType.Angle:
                    turnOverAngleCoroutine = StartCoroutine(TurnOverAngle(Mathf.Clamp(angle, FullTurnAngle.min, FullTurnAngle.max)));
                    break;
            }

        }

        public void UseSkill()
        {
            // Debug.Log($"isSkillDisabled {isSkillDisabled}");

            if (isSkillDisabled) return;
            Skill.Activate(Skill.Type);
        }

        IEnumerator TurnOverAngle(float totalAngle)
        {
            // Initially call the debounce-logger before turning logic starts
            if (totalAngle < 0)
            {
                LogManager.CallPlayerActionLog(Side, "TurnLeft", param: totalAngle.ToString());
            }
            else
            {
                LogManager.CallPlayerActionLog(Side, "TurnRight", param: totalAngle.ToString());
            }

            float rotatedAngle = 0f;

            //Duration based on speed of robot
            float duration = Mathf.Abs(totalAngle) * TurnRate / LastVelocity.magnitude;
            if (duration > TurnRate)
            {
                duration = TurnRate;
            }

            float speed = totalAngle / duration; // degrees per second

            // Start turning with [duration]
            while (Mathf.Abs(rotatedAngle) < Mathf.Abs(totalAngle) && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing && !IsMovementLocked)
            {
                float delta = speed * Time.deltaTime; // how much to rotate this frame

                if (Mathf.Abs(rotatedAngle + delta) > Mathf.Abs(totalAngle))
                {
                    delta = totalAngle - rotatedAngle; // clamp to finish exactly
                }

                transform.Rotate(0, 0, delta); // rotate
                rotatedAngle += delta; // track how much we've rotated

                // Keep calling debounce-logger until [duration] satisfied
                if (totalAngle < 0)
                {
                    LogManager.CallPlayerActionLog(Side, "TurnLeft");
                }
                else
                {
                    LogManager.CallPlayerActionLog(Side, "TurnRight");
                }

                yield return null;
            }
        }

        IEnumerator AccelerateOverTime(float time, bool isDash = false)
        {
            float elapsedTime = 0f;
            float speed = isDash ? DashSpeed : MoveSpeed;

            // Initially call debounce-logger before accelerating logic starts
            if (isDash)
            {
                LogManager.CallPlayerActionLog(Side, "Dash", param: time.ToString());
            }
            else
            {
                LogManager.CallPlayerActionLog(Side, "Accelerate", param: time.ToString());
            }

            // lerping?, uncomment
            // Vector2 initialVelocity = robotRigidBody.linearVelocity; 

            // Start accelerating with [time]
            while (elapsedTime < time && BattleManager.Instance.CurrentState == BattleState.Battle_Ongoing && !IsMovementLocked)
            {
                // lerping?, uncomment
                // float t = elapsedTime / time;
                // float currentSpeed = Mathf.Lerp(initialVelocity.magnitude, targetSpeed, t);

                robotRigidBody.linearVelocity = transform.up.normalized * speed;

                elapsedTime += Time.deltaTime;

                // Keep calling debounce-logger until [time] satisfied
                if (isDash)
                {
                    LogManager.CallPlayerActionLog(Side, "Dash");
                }
                else
                {
                    LogManager.CallPlayerActionLog(Side, "Accelerate");
                }
                yield return null;
            }

            // robotRigidBody.linearVelocity = Vector2.Lerp(robotRigidBody.linearVelocity, Vector2.zero, SlowDownRate * Time.deltaTime);
        }
        #endregion


        #region Robot Movement & Skill State
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

        #region Robot Physics

        void BounceRule(Collision2D collision)
        {
            if (!Application.isPlaying) return;

            var otherRobot = collision.gameObject.GetComponent<SumoController>();
            if (otherRobot == null) return;

            // Myspeed
            float actorVelocity = LastVelocity.magnitude;
            float targetVelocity = otherRobot.LastVelocity.magnitude;

            StopCoroutineAction();
            otherRobot.StopCoroutineAction();


            // Only the faster one handles the bounce
            if (actorVelocity >= targetVelocity)
            {

                Vector2 collisionNormal = collision.contacts[0].normal;

                float total = actorVelocity + targetVelocity + 0.01f;

                if (total < 0.1f) return;

                float actorImpact = CollisionBaseForce * targetVelocity / total;  // robotA gets more bounce if B has more speed
                float targetImpact = CollisionBaseForce * actorVelocity / total;  // robotB gets more bounce if A has more speed

                // Check if Actor using Stone, then calculate the Target impact
                if (Skill.Type == ERobotSkillType.Stone && Skill.IsActive)
                {
                    targetImpact = targetVelocity / total;
                }

                // Check if Target using Stone, then calculate the Actor impact
                if (otherRobot.Skill.Type == ERobotSkillType.Stone && otherRobot.Skill.IsActive)
                {
                    actorImpact = actorVelocity / total;
                }

                // Applying opposite bounce back multiplier
                actorImpact *= otherRobot.BounceResistance;
                targetImpact *= BounceResistance;


                Bounce(collisionNormal, actorImpact);       // away from B
                otherRobot.Bounce(-collisionNormal, targetImpact);      // away from A

                // Inform listener that robot get bounce
                OnPlayerBounce?.Invoke(Side);
                otherRobot.OnPlayerBounce?.Invoke(Side);

                float actorLockDuration = LockMovement(isActor: true, actorImpact);
                float targetLockDuration = otherRobot.LockMovement(isActor: false, targetImpact);

                Debug.Log($"[BounceRule]\nActor=>{Side},Target=>{otherRobot.Side}\nActorVelocity=>{actorVelocity},TargetVelocity=>{targetVelocity}\nActorCurrentSkill=>{Skill.Type}, TargetCurrentSkill=>{otherRobot.Skill.Type}\nActorImpact=>{actorImpact}, TargetImpact=>{targetImpact}");

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
                // Gradually decrease linear and angular velocities 
                robotRigidBody.linearVelocity = Vector2.Lerp(robotRigidBody.linearVelocity, Vector2.zero, SlowDownRate * Time.deltaTime); //[Todo] Need to just stop after it close to zero.
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

        #region Robot Appearance
        public void UpdateFaceColor()
        {
            if (Side == PlayerSide.Left)
            {
                face.color = new Color(0, 255, 0);
            }
            else
            {
                face.color = new Color(255, 0, 0);
            }
        }
        #endregion
    }
}