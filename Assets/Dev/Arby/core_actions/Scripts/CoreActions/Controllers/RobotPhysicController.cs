using System;
using UnityEngine;

namespace RobotCoreAction
{
    namespace Controllers
    {
        public class RobotPhysicController : MonoBehaviour
        {
            private RobotStats stats;
            private float lastActTime = 0.0f;        // Last input or collision time, excluding rotation.
            private Rigidbody2D rigidBody;

            public Vector2 LastVelocity { get; private set; } = Vector2.zero;

            public event Action<Collision2D> OnColisionEvents;


            private void Awake()
            {
                stats = GetComponent<RobotStats>();
                rigidBody = GetComponent<Rigidbody2D>();
                rigidBody.gravityScale = 0;        // Disable gravity for top-down movement
                rigidBody.linearDamping = 0;       // No drag, we handle it manually
                rigidBody.angularDamping = 0;      // No automatic angular slow-down
            }

            private void Update()
            {
                UpdateDashState();

            }

            private void FixedUpdate()
            {
                HandleStopping();
            }

            void OnCollisionEnter2D(Collision2D collision)
            {
                OnColisionEvents?.Invoke(collision);
            }

            private void UpdateDashState()
            {
                LastVelocity = rigidBody.linearVelocity;
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
                    rigidBody.linearVelocity = Vector2.Lerp(rigidBody.linearVelocity, Vector2.zero, stats.SlowDownRate * Time.deltaTime); //[Todo] Need to just stop after it close to zero.
                    rigidBody.angularVelocity = Mathf.Lerp(rigidBody.angularVelocity, 0, stats.SlowDownRate * Time.deltaTime);
                }
            }

            public void Bounce(Vector2 direction, float force)
            {
                rigidBody.linearVelocity = direction * force;
                //[Todo] Need to handle after get hit from dashing enemy, uncontrollable for short time. 
            }

            public void FreezeMovement()
            {
                rigidBody.constraints = RigidbodyConstraints2D.FreezePosition;
            }

            public void ResetFreezeMovement()
            {
                rigidBody.constraints = RigidbodyConstraints2D.None;
            }

            public void SetLastVelocity(Vector2 value)
            {
                LastVelocity = value;
            }

        }
    }

}