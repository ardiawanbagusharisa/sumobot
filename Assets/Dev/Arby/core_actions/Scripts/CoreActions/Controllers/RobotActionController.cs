using RobotCoreAction.Skills;
using UnityEngine;

namespace RobotCoreAction
{
    namespace Controllers
    {
        public class RobotActionController : MonoBehaviour
        {
            private RobotStats stats;
            private RobotPhysicController physic;
            private Rigidbody2D robotRigidBody;

            private void Awake()
            {
                stats = GetComponent<RobotStats>();
                physic = GetComponent<RobotPhysicController>();
                robotRigidBody = GetComponent<Rigidbody2D>();
            }

            public void Accelerate()
            {
                if (stats.isMoveDisabled)
                {
                    Debug.Log("Move is disabled.");
                    return;
                }

                float speed = stats.LastRobotActionType == ERobotActionType.Dash ? stats.DashSpeed : stats.MoveSpeed; //[Todo] This could be redundant with the Dash(). 
                robotRigidBody.linearVelocity = transform.up * speed;

            }

            public void Dash()
            {
                if (stats.isMoveDisabled)
                {
                    Debug.Log("Move is disabled.");
                    return;
                }

                stats.LastRobotActionType = ERobotActionType.Dash;
                robotRigidBody.linearVelocity = transform.up * stats.DashSpeed;
            }

            public void Turn(bool isRight)
            {
                float rotation = (isRight ? -stats.RotateSpeed : stats.RotateSpeed) * Time.deltaTime;
                transform.Rotate(0, 0, rotation);
            }

            public void UseSkill(ISkill skill)
            {
                skill.Execute(this, stats, physic);
            }
        }
    }
}