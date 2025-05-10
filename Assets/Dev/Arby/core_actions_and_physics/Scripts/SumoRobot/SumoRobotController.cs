using System;
using System.Collections;
using System.Collections.Generic;
using BattleLoop;
using UnityEngine;

namespace CoreSumoRobot
{
    public enum PlayerSide
    {
        Left,
        Right,
    }

    public class SumoRobotController : MonoBehaviour
    {
        #region Basic Stats
        public int IdInt;
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
        public float BounceResistance = 1f;

        public PlayerSide Side => IdInt == 0 ? PlayerSide.Left : PlayerSide.Right;
        #endregion

        public Vector2 LastVelocity { get; private set; } = Vector2.zero;
        public bool isInputDisabled = false;
        public SpriteRenderer face;
        public Transform StartPosition;
        public event Action<PlayerSide> OnOutOfArena;
        public SumoSkill sumoSkill;

        private bool isMoveDisabled = false;
        private bool isSkillDisabled = false;
        private float reservedMoveSpeed;
        private float reservedDashSpeed;
        private float reserverdBounceResistance;
        private Rigidbody2D robotRigidBody;
        private InputProvider provider;
        private float moveLockTime = 0f;
        private bool IsMovementLocked => moveLockTime > 0f;

        private event Action<Collision2D> onEnterCollisions;
        private event Action<Collider2D> onExitTriggers;
        private bool hasOnOutOfArenaInvoked = false;

        private void Awake()
        {
            sumoSkill = new SumoSkill(this);
            robotRigidBody = GetComponent<Rigidbody2D>();
            reservedMoveSpeed = MoveSpeed;
            reservedDashSpeed = DashSpeed;
            reserverdBounceResistance = BounceResistance;
            SetRules(true);
        }

        void OnDestroy()
        {
            SetRules(false);
        }

        private void Update()
        {
            ReadInput();
            UpdateDashState();
        }

        private void FixedUpdate()
        {
            HandleStopping();
            if (moveLockTime > 0f)
                moveLockTime -= Time.deltaTime;
        }
        void OnCollisionEnter2D(Collision2D collision)
        {
            onEnterCollisions?.Invoke(collision);
        }

        void OnTriggerExit2D(Collider2D collision)
        {
            onExitTriggers?.Invoke(collision);
        }


        #region Robot State

        public void IsInArena(Collider2D collider)
        {
            if (collider.tag == "Arena/Floor" && !hasOnOutOfArenaInvoked)
            {
                OnOutOfArena?.Invoke(Side);
                hasOnOutOfArenaInvoked = true;
            }
        }

        private void SetRules(bool isEnabled)
        {
            if (isEnabled)
            {
                onEnterCollisions += BounceRule;
                onExitTriggers += IsInArena;
            }
            else
            {
                onEnterCollisions -= BounceRule;
                onExitTriggers -= IsInArena;
            }
        }

        public void ResetForNewBattle()
        {
            transform.position = StartPosition.position;
            transform.rotation = StartPosition.rotation;
            hasOnOutOfArenaInvoked = false;
            ResetActionData();
        }
        #endregion


        #region Robot Action Data
        public Dictionary<ERobotActionType, float> ActionsTime = new Dictionary<ERobotActionType, float>();

        private ERobotActionType _lastRobotActionType = ERobotActionType.Idle;

        public ERobotActionType LastRobotActionType
        {
            get { return _lastRobotActionType; }
            set
            {
                ActionsTime[_lastRobotActionType] = Time.time;
                _lastRobotActionType = value;
            }
        }

        private void ResetActionData()
        {
            ActionsTime = new Dictionary<ERobotActionType, float>();
            LastRobotActionType = ERobotActionType.Idle;

            sumoSkill.Reset();
        }
        #endregion


        #region Robot Action
        public void Accelerate(AccelerateActionType type, float time = float.NaN)
        {
            if (isMoveDisabled || IsMovementLocked)
            {
                Debug.Log("Move is disabled.");
                return;
            }

            switch (type)
            {
                case AccelerateActionType.Default:
                    robotRigidBody.linearVelocity = transform.up * MoveSpeed;
                    break;
                case AccelerateActionType.Time:
                    if (time == float.NaN) throw new Exception("Time can't be NaN when you are using [AccelerateActionType.Time] type");
                    StartCoroutine(AccelerateOverTime(time));
                    break;
            }
        }

        public void Dash(DashActionType type, float time = float.NaN)
        {
            if (isMoveDisabled)
            {
                Debug.Log("Move is disabled.");
                return;
            }

            BattleManager.Instance.CurrentRound.SetActionLog(Side, $"type=action;dash={type},{time};robotId={IdInt}");

            switch (type)
            {
                case DashActionType.Default:
                    ActionsTime.TryGetValue(ERobotActionType.Dash, out float lastActTime);
                    if (Time.time >= lastActTime + DashDuration)
                    {
                        LastRobotActionType = ERobotActionType.Dash;
                        robotRigidBody.linearVelocity = transform.up * DashSpeed;
                    }
                    else
                    {
                        Debug.Log("Dash is on cooldown.");
                    }
                    break;
                case DashActionType.Time:
                    if (time == float.NaN) throw new Exception("Time can't be NaN when you are using [DashActionType.Time] type");
                    StartCoroutine(AccelerateOverTime(time, isDash: true));
                    break;
            }

        }

        public void Turn(TurnActionType type = TurnActionType.Angle, float angle = float.NaN)
        {
            BattleManager.Instance.CurrentRound.SetActionLog(Side, $"type=action;turn={type},{angle};robotId={IdInt}");

            switch (type)
            {
                case TurnActionType.Left:
                    transform.Rotate(0, 0, RotateSpeed * Time.deltaTime);
                    break;
                case TurnActionType.Right:
                    transform.Rotate(0, 0, -RotateSpeed * Time.deltaTime);
                    break;
                case TurnActionType.LeftAngle:
                    if (angle == float.NaN) throw new Exception("Left Angle can't be NaN when you are using [TurnActionInput.LeftAngle] type");
                    if (angle < 0) throw new Exception("Left Angle can't be < 0 when you are using [TurnActionInput.LeftAngle] type");
                    StartCoroutine(TurnOverAngle(angle * 1f, 0.5f));
                    break;
                case TurnActionType.RightAngle:
                    if (angle == float.NaN) throw new Exception("Right Angle can't be NaN when you are using [TurnActionInput.RightAngle] type");
                    if (angle < 0) throw new Exception("Right Angle can't be < 0 when you are using [TurnActionInput.RightAngle] type");
                    StartCoroutine(TurnOverAngle(angle * -1f, 0.5f));
                    break;
                case TurnActionType.Angle:
                    if (angle == float.NaN) throw new Exception("Angle can't be NaN when you are using [TurnActionInput.Angle] type");
                    StartCoroutine(TurnOverAngle(angle, 0.5f));
                    break;
            }

        }

        public void UseSkill(ERobotSkillType skillType)
        {
            Debug.Log($"isSkillDisabled {isSkillDisabled}");

            if (isSkillDisabled) return;
            sumoSkill.Activate(skillType);
        }

        IEnumerator TurnOverAngle(float totalAngle, float duration)
        {
            BattleManager.Instance.CurrentRound.SetActionLog(Side, $"type=action;turn_angle={totalAngle},{duration};robotId={IdInt}");

            float rotatedAngle = 0f;
            float speed = totalAngle / duration; // degrees per second

            while (Mathf.Abs(rotatedAngle) < Mathf.Abs(totalAngle))
            {
                float delta = speed * Time.deltaTime; // how much to rotate this frame

                if (Mathf.Abs(rotatedAngle + delta) > Mathf.Abs(totalAngle))
                {
                    delta = totalAngle - rotatedAngle; // clamp to finish exactly
                }

                transform.Rotate(0, 0, delta); // rotate
                rotatedAngle += delta; // track how much we've rotated

                yield return null;
            }
        }

        IEnumerator AccelerateOverTime(float time, bool isDash = false)
        {
            BattleManager.Instance.CurrentRound.SetActionLog(Side, $"type=action;accelerate_time={time},{isDash};robotId={IdInt}");

            float elapsedTime = 0f;
            float speed = isDash ? DashSpeed : MoveSpeed;

            // lerping?, uncomment
            // Vector2 initialVelocity = robotRigidBody.linearVelocity; 

            while (elapsedTime < time)
            {
                // lerping?, uncomment
                // float t = elapsedTime / time;
                // float currentSpeed = Mathf.Lerp(initialVelocity.magnitude, targetSpeed, t);

                robotRigidBody.linearVelocity = transform.up.normalized * speed;

                elapsedTime += Time.deltaTime;
                yield return null;
            }

            robotRigidBody.linearVelocity = Vector2.Lerp(robotRigidBody.linearVelocity, Vector2.zero, SlowDownRate * Time.deltaTime);
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

        public void ChangeMoveSpeed(float value)
        {
            MoveSpeed = value;
        }

        public void ChangeDashSpeed(float value)
        {
            DashSpeed = value;
        }

        public void ResetMoveSpeed()
        {
            MoveSpeed = reservedMoveSpeed;
        }

        public void ResetDashSpeed()
        {
            DashSpeed = reservedDashSpeed;
        }

        public void LockMovement(float duration)
        {
            moveLockTime = Mathf.Max(moveLockTime, duration);
        }
        #endregion

        #region Robot Physics

        void BounceRule(Collision2D collision)
        {
            var otherRobot = collision.gameObject.GetComponent<SumoRobotController>();
            if (otherRobot == null) return;

            // Compare magnitudes
            float mySpeed = LastVelocity.magnitude;
            float otherSpeed = otherRobot.LastVelocity.magnitude;

            // Only the faster one handles the bounce
            if (mySpeed >= otherSpeed)
            {
                Vector2 collNormal = collision.contacts[0].normal;
                PhysicHelper.HandleBounce(this, otherRobot, collNormal);
            }
        }

        public void Bounce(Vector2 direction, float force)
        {
            float lockDuration = Mathf.Clamp(force * 0.5f, 1f, 2.5f);
            Debug.Log("Bounce.LockDuration" + lockDuration);
            LockMovement(lockDuration);

            robotRigidBody.linearVelocity = direction * force;
        }

        public void FreezeMovement()
        {
            robotRigidBody.constraints = RigidbodyConstraints2D.FreezePosition;
        }

        public void ResetFreezeMovement()
        {
            robotRigidBody.constraints = RigidbodyConstraints2D.None;
        }

        public void SetLastVelocity(Vector2 value)
        {
            LastVelocity = value;
        }

        public void SetBounceResistance(float value)
        {
            BounceResistance = value;
        }


        public void ResetBounceResistance()
        {
            BounceResistance = reserverdBounceResistance;
        }

        private void UpdateDashState()
        {
            LastVelocity = robotRigidBody.linearVelocity;
            ActionsTime.TryGetValue(ERobotActionType.Dash, out float lastActTime);
            if (LastRobotActionType == ERobotActionType.Dash && Time.time >= lastActTime + DashDuration)
            {
                LastRobotActionType = ERobotActionType.Idle;
            }
        }

        private void HandleStopping()
        {
            ActionsTime.TryGetValue(ERobotActionType.Dash, out float lastActTime);
            if (Time.time > lastActTime + StopDelay)
            {
                // Gradually decrease linear and angular velocities 
                robotRigidBody.linearVelocity = Vector2.Lerp(robotRigidBody.linearVelocity, Vector2.zero, SlowDownRate * Time.deltaTime); //[Todo] Need to just stop after it close to zero.
                robotRigidBody.angularVelocity = Mathf.Lerp(robotRigidBody.angularVelocity, 0, SlowDownRate * Time.deltaTime);
            }
        }
        #endregion

        #region Robot Movement Input

        public void UseInput(InputProvider inputProvider)
        {
            provider = inputProvider;
        }
        void ReadInput()
        {
            if (provider == null) return;
            if (isInputDisabled) return;

            List<ISumoAction> actions = provider.GetInput();
            foreach (ISumoAction action in actions)
            {
                action.Execute(this);
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