using System.Collections;
using UnityEngine;

namespace RobotCoreAction
{

    public interface ISkill
    {
        ERobotSkillType SkillType { get; }
        float Duration { get; }
        float Cooldown { get; }
        void Execute(CoreActionRobotController controller, CoreActionRobot robot);
    }
    public enum ERobotSkillType
    {
        Stone,
        Boost
    }

    public class BoostSkill : ISkill
    {
        #region Boost Skill Stat
        public ERobotSkillType SkillType => ERobotSkillType.Boost;
        public float Duration => 5f;
        public float Cooldown => 10f;
        public float BoostMultiplier => 1.8f;
        #endregion

        private CoreActionRobot robot;
        private CoreActionRobotController controller;


        public void Execute(CoreActionRobotController controllerParam, CoreActionRobot robotParam)
        {
            OnExecute(controllerParam, robotParam);
        }

        private void OnExecute(CoreActionRobotController controllerParam, CoreActionRobot robotParam)
        {
            controller = controllerParam;
            robot = robotParam;
            controller.SkillTime.TryGetValue(SkillType, out float lastSkillTime);
            if (Time.time >= lastSkillTime + Cooldown)
            {
                //Debug
                SkillCooldownUI.Instance.ShowSkillCooldown(this);

                Activate();

                controller.LastRobotActionType = ERobotActionType.Skill;
                controller.SkillTime[SkillType] = Time.time; // Update the last skill time
            }
            else
            {
                Debug.Log("[Skill][Boost] on cooldown.");
            }
        }

        private void Activate()
        {
            Debug.Log("[Skill][Boost] activated!");
            controller.EnableMove(); // Enable movement
            controller.ChangeMoveSpeed(robot.MoveSpeed * BoostMultiplier);
            controller.ChangeDashSpeed(robot.DashSpeed * BoostMultiplier);

            controller.StartCoroutine(DeactivateAfterDuration());
        }

        public IEnumerator DeactivateAfterDuration()
        {
            yield return new WaitForSeconds(Duration);
            Deactivate();
        }

        private void Deactivate()
        {
            controller.ResetMoveSpeed();
            controller.ResetDashSpeed();
            Debug.Log("[Skill][Boost] deactivated!");
        }
    }

    public class StoneSkill : ISkill
    {
        #region Stone Skill Stat
        public ERobotSkillType SkillType => ERobotSkillType.Stone;
        public float Duration => 5f;
        public float Cooldown => 10f;
        public float BounceBackMultiplier => 1.2f;
        #endregion

        private CoreActionRobot robot;
        private CoreActionRobotController controller;



        public void Execute(CoreActionRobotController controllerParam, CoreActionRobot robotParam)
        {
            robot = robotParam;
            controller = controllerParam;
            controller.SkillTime.TryGetValue(SkillType, out float lastSkillTime);
            if (Time.time >= lastSkillTime + Cooldown)
            {
                //Debug
                SkillCooldownUI.Instance.ShowSkillCooldown(this);


                Activate();

                controller.LastRobotActionType = ERobotActionType.Skill;
                controller.SkillTime[SkillType] = Time.time; // Update the last skill time
            }
            else
            {
                Debug.Log("[Skill][Stone] on cooldown.");
            }
        }

        private void Activate()
        {
            controller.FreezeMovement();
            controller.DisableMove(); // Disable movement
            controller.OnColisionEvents += OnCollision; // Subscribe to collision events
            Debug.Log("[Skill][Stone] activated!");

            controller.StartCoroutine(DeactivateAfterDuration());

        }

        public IEnumerator DeactivateAfterDuration()
        {
            yield return new WaitForSeconds(Duration);
            Deactivate();
        }

        public void OnCollision(Collision2D collision)
        {
            // Implement the logic for what happens when the robot collides with something while the skill is active
            Debug.Log("Collision detected on " + collision.gameObject.layer + " during stone skill!");

            var enemyController = collision.gameObject.GetComponent<CoreActionRobotController>();

            Vector2 collisionNormal = collision.contacts[0].normal; //[Todo] GetContact()
            Vector2 bounceDirection = Vector2.Reflect(enemyController.LastVelocity.normalized, collisionNormal);

            float impactForce = enemyController.LastVelocity.magnitude * BounceBackMultiplier;
            enemyController.Bounce(bounceDirection, impactForce);
        }

        private void Deactivate()
        {
            // Implement the logic to deactivate the skill
            Debug.Log("[Skill][Stone] deactivated!");
            controller.ResetFreezeMovement();
            controller.OnColisionEvents -= OnCollision; // Unsubscribe from collision events
            controller.EnableMove(); // Enable movement
        }

    }

}