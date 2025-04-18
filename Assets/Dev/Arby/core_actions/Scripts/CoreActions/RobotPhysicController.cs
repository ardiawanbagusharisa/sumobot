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
        private Rigidbody2D rb;

        public Vector2 LastVelocity { get; private set; } = Vector2.zero;

        private void Awake()
        {
            stats = GetComponent<RobotStats>();
            rb = GetComponent<Rigidbody2D>();
            rb.gravityScale = 0;        // Disable gravity for top-down movement
            rb.linearDamping = 0;       // No drag, we handle it manually
            rb.angularDamping = 0;      // No automatic angular slow-down
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
            LastVelocity = rb.linearVelocity;
            if (stats.LastRobotActionType == ERobotActionType.Dash && Time.time >= lastActTime + stats.DashDuration)
            {
                stats.LastRobotActionType = ERobotActionType.Idle;
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

        public void Bounce(Vector2 direction, float force)
        {
            rb.linearVelocity = direction * force;
            //[Todo] Need to handle after get hit from dashing enemy, uncontrollable for short time. 
        }

        public void FreezeMovement()
        {
            rb.constraints = RigidbodyConstraints2D.FreezePosition;
        }

        public void ResetFreezeMovement()
        {
            rb.constraints = RigidbodyConstraints2D.None;
        }

        public void SetLastVelocity(Vector2 value)
        {
            LastVelocity = value;
        }

    }
}