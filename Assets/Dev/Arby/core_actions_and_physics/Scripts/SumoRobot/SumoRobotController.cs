using System;
using System.Collections.Generic;
using UnityEditor.Rendering;
using UnityEngine;

namespace CoreSumoRobot
{
    public class SumoRobotController : MonoBehaviour
    {
        public Vector2 LastVelocity { get; private set; } = Vector2.zero;
        public bool isInputDisabled = false;
        public SpriteRenderer face;
        public Transform StartPosition;
        public event Action<int> OnOutOfArena;

        private bool isMoveDisabled = false;
        private SumoRobot sumoRobot;
        private float reservedMoveSpeed;
        private float reservedDashSpeed;
        private float reserverdBounceResistance;
        private Rigidbody2D robotRigidBody;
        private IInputProvider provider;
        private float moveLockTime = 0f;
        private bool IsMovementLocked => moveLockTime > 0f;

        private event Action<Collision2D> onEnterCollisions;
        private event Action<Collider2D> onExitTriggers;

        private void Awake()
        {
            sumoRobot = GetComponent<SumoRobot>();
            robotRigidBody = GetComponent<Rigidbody2D>();
            reservedMoveSpeed = sumoRobot.MoveSpeed;
            reservedDashSpeed = sumoRobot.DashSpeed;
            reserverdBounceResistance = sumoRobot.BounceResistance;
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
            if (collider.tag == "Arena/Floor")
            {
                OnOutOfArena?.Invoke(sumoRobot.IdInt);
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

        public void Reset()
        {
            transform.position = StartPosition.position;
            transform.rotation = StartPosition.rotation;
            ResetActionData();
        }
        #endregion


        #region Robot Action Data
        public Dictionary<ERobotActionType, float> ActionsTime = new Dictionary<ERobotActionType, float>();

        public Dictionary<ERobotSkillType, float> SkillTime = new Dictionary<ERobotSkillType, float>();

        private ERobotActionType _lastRobotActionType = ERobotActionType.Idle;


        private ERobotSkillType _lastRobotSkillType;
        public ERobotSkillType LastRobotSkillType
        {
            get { return _lastRobotSkillType; }
            set
            {
                _lastRobotSkillType = value;
                _lastRobotActionType = ERobotActionType.Skill;

                if (value != ERobotSkillType.None)
                {
                    SkillTime[value] = Time.time;
                }
            }
        }

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
            SkillTime = new Dictionary<ERobotSkillType, float>();
            LastRobotActionType = ERobotActionType.Idle;
            LastRobotSkillType = ERobotSkillType.None;
        }
        #endregion


        #region Robot Action
        public void Accelerate()
        {
            if (isMoveDisabled || IsMovementLocked)
            {
                Debug.Log("Move is disabled.");
                return;
            }

            float speed = LastRobotActionType == ERobotActionType.Dash ? sumoRobot.DashSpeed : sumoRobot.MoveSpeed; //[Todo] This could be redundant with the Dash(). 
            robotRigidBody.linearVelocity = transform.up * speed;

        }

        public void Dash()
        {
            if (isMoveDisabled)
            {
                Debug.Log("Move is disabled.");
                return;
            }

            ActionsTime.TryGetValue(ERobotActionType.Dash, out float lastActTime);
            if (Time.time >= lastActTime + sumoRobot.DashDuration)
            {

                LastRobotActionType = ERobotActionType.Dash;
                robotRigidBody.linearVelocity = transform.up * sumoRobot.DashSpeed;
            }
            else
            {
                Debug.Log("Dash is on cooldown.");
            }

        }

        public void Turn(bool isRight)
        {
            float rotation = (isRight ? -sumoRobot.RotateSpeed : sumoRobot.RotateSpeed) * Time.deltaTime;
            transform.Rotate(0, 0, rotation);
        }

        public void UseSkill(ISkill skill)
        {
            skill.Execute(this, sumoRobot);
        }
        #endregion


        #region Robot Movement State
        public void SetMovementEnabled(bool value)
        {
            isMoveDisabled = !value;
        }
        public void SetInputEnabled(bool value)
        {
            isInputDisabled = !value;
        }

        public void ChangeMoveSpeed(float value)
        {
            sumoRobot.MoveSpeed = value;
        }


        public void ChangeDashSpeed(float value)
        {
            sumoRobot.DashSpeed = value;
        }

        public void ResetMoveSpeed()
        {
            sumoRobot.MoveSpeed = reservedMoveSpeed;
        }

        public void ResetDashSpeed()
        {
            sumoRobot.DashSpeed = reservedDashSpeed;
        }

        public void LockMovement(float duration)
        {
            moveLockTime = Mathf.Max(moveLockTime, duration);
        }
        #endregion

        #region Robot Physics

        void BounceRule(Collision2D collision)
        {
            var collNormal = collision.contacts[0].normal;

            PhysicHelper.HandleBounce(this, collision.gameObject.GetComponent<SumoRobotController>(), collNormal);
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
            sumoRobot.BounceResistance = value;
        }


        public void ResetBounceResistance()
        {
            sumoRobot.BounceResistance = reserverdBounceResistance;
        }

        private void UpdateDashState()
        {
            LastVelocity = robotRigidBody.linearVelocity;
            ActionsTime.TryGetValue(ERobotActionType.Dash, out float lastActTime);
            if (LastRobotActionType == ERobotActionType.Dash && Time.time >= lastActTime + sumoRobot.DashDuration)
            {
                LastRobotActionType = ERobotActionType.Idle;
            }
        }

        private void HandleStopping()
        {
            ActionsTime.TryGetValue(ERobotActionType.Dash, out float lastActTime);
            if (Time.time > lastActTime + sumoRobot.StopDelay)
            {
                // Gradually decrease linear and angular velocities 
                robotRigidBody.linearVelocity = Vector2.Lerp(robotRigidBody.linearVelocity, Vector2.zero, sumoRobot.SlowDownRate * Time.deltaTime); //[Todo] Need to just stop after it close to zero.
                robotRigidBody.angularVelocity = Mathf.Lerp(robotRigidBody.angularVelocity, 0, sumoRobot.SlowDownRate * Time.deltaTime);
            }
        }
        #endregion

        #region Robot Movement Input

        public void UseInput(IInputProvider inputProvider)
        {
            provider = inputProvider;
        }
        void ReadInput()
        {
            if (provider == null) return;
            if (isInputDisabled) return;

            var actions = provider.GetInput();
            foreach (var action in actions)
            {
                action.Execute(this);
            }
        }
        #endregion

        #region Robot Appearance
        public void ChangeFaceColor(bool isLeftSide)
        {
            if (isLeftSide)
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