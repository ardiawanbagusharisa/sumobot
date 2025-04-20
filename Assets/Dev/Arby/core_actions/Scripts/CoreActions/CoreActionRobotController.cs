using System;
using System.Collections.Generic;
using UnityEngine;

namespace RobotCoreAction
{
    public class CoreActionRobotController : MonoBehaviour
    {
        public Vector2 LastVelocity { get; private set; } = Vector2.zero;
        public event Action<Collision2D> OnColisionEvents;
        public bool isMoveDisabled = false;
        public bool isInputDisabled = false;
        private CoreActionRobot sumoRobot;
        private float reservedMoveSpeed;
        private float reservedDashSpeed;
        private Rigidbody2D robotRigidBody;
        private IInputProvider provider;

        private void Awake()
        {
            sumoRobot = GetComponent<CoreActionRobot>();
            robotRigidBody = GetComponent<Rigidbody2D>();
            reservedMoveSpeed = sumoRobot.MoveSpeed;
            reservedDashSpeed = sumoRobot.DashSpeed;
        }
        private void Update()
        {
            ReadInput();
            UpdateDashState();
        }

        private void FixedUpdate()
        {
            HandleStopping();
        }


        #region Robot Action Data
        public Dictionary<ERobotActionType, float> ActionsTime = new Dictionary<ERobotActionType, float>();

        public Dictionary<ERobotSkillType, float> SkillTime = new Dictionary<ERobotSkillType, float>();

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
        #endregion


        #region Robot Action
        public void Accelerate()
        {
            if (isMoveDisabled)
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
        public void DisableMove()
        {
            isMoveDisabled = true;
        }

        public void EnableMove()
        {
            isMoveDisabled = false;
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
        #endregion

        #region Robot Physics
        void OnCollisionEnter2D(Collision2D collision)
        {
            OnColisionEvents?.Invoke(collision);
        }

        public void Bounce(Vector2 direction, float force)
        {
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
    }
}