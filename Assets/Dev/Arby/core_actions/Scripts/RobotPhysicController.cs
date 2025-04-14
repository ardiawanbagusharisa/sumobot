using System;
using System.Collections.Generic;
using RobotCoreAction;
using UnityEngine;
namespace RobotCoreAction
{
    public class RobotPhysicController : MonoBehaviour
    {
        private RobotStats stats;
        private float lastActTime = 0.0f;        // Last input or collision time, excluding rotation.
        private ERobotActionType _lastRobotActionType = ERobotActionType.Idle;
        private Rigidbody2D rb;


        private void Awake()
        {
            stats = GetComponent<RobotStats>();
            rb = GetComponent<Rigidbody2D>();
            rb.gravityScale = 0;        // Disable gravity for top-down movement
            rb.linearDamping = 0;       // No drag, we handle it manually
            rb.angularDamping = 0;      // No automatic angular slow-down
        }

        public ERobotActionType LastRobotActionType
        {
            get { return _lastRobotActionType; }
            set
            {
                stats.ActionsTime[_lastRobotActionType] = Time.time;
                _lastRobotActionType = value;
                lastActTime = Time.time;
            }
        }

        private void Update()
        {
            UpdateDashState();

        }

        private void FixedUpdate()
        {
            HandleStopping();            
        }

        private void UpdateDashState()
        {
            if (LastRobotActionType == ERobotActionType.Dash && Time.time >= lastActTime + stats.DashDuration)
            {
                LastRobotActionType = ERobotActionType.Idle;
            }
        }

        private void HandleStopping()
        {
            if (Time.time > lastActTime + stats.StopDelay)
            {
                // Gradually decrease linear and angular velocities 
                rb.linearVelocity = Vector2.Lerp(rb.linearVelocity, Vector2.zero, stats.SlowDownRate * Time.deltaTime); //[Todo] Need to just stop after it close to zero.
                rb.angularVelocity = Mathf.Lerp(rb.angularVelocity, 0, stats.SlowDownRate * Time.deltaTime);
            }
        }

    }
}