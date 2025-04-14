using UnityEngine;

namespace RobotCoreAction
{
    public class RobotActionController : MonoBehaviour
    {
        private RobotStats stats;
        private RobotPhysicController robotController;
        private Rigidbody2D _robotRigidBody;

        private void Awake()
        {
            stats = GetComponent<RobotStats>();
            robotController = GetComponent<RobotPhysicController>();
            _robotRigidBody = GetComponent<Rigidbody2D>();
        }

        public void Accelerate()
        {
            float speed = robotController.LastRobotActionType == ERobotActionType.Dash ? stats.DashSpeed : stats.MoveSpeed; //[Todo] This could be redundant with the Dash(). 
            _robotRigidBody.linearVelocity = transform.up * speed;

        }

        public void Dash()
        {
            robotController.LastRobotActionType = ERobotActionType.Dash;
            _robotRigidBody.linearVelocity = transform.up * stats.DashSpeed;
        }

        public void Turn(bool isRight)
        {
            float rotation = (isRight ? -stats.RotateSpeed : stats.RotateSpeed) * Time.deltaTime;
            transform.Rotate(0, 0, rotation);
        }

        public void UseSkill()
        {
            robotController.LastRobotActionType = ERobotActionType.Skill;

            // Trigger skill logic
        }
    }
}