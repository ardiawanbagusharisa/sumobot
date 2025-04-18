using System;
using System.Collections;
using Unity.VisualScripting.Antlr3.Runtime.Misc;
using UnityEngine;

namespace RobotCoreAction
{
    public class StoneSkill : MonoBehaviour, ISkill
    {

        private RobotActionController robotActionController;
        private RobotStats robotStats;
        private RobotPhysicController robotPhysic;

        public float Duration => 5f;
        public float Cooldown => 10f;
        public float BounceBackMultiplier => 1.2f;

        public ERobotSkillType SkillType => ERobotSkillType.Stone;

        public void Execute(RobotActionController controller, RobotStats stats, RobotPhysicController physic)
        {
            robotPhysic = physic;
            robotActionController = controller;
            robotStats = stats;
            stats.SkillTime.TryGetValue(SkillType, out float lastSkillTime);
            if (Time.time >= lastSkillTime + Cooldown)
            {
                //Debug
                SkillCooldownUI.Instance.ShowSkillCooldown(this);


                Activate();

                stats.LastRobotActionType = ERobotActionType.Skill;
                stats.SkillTime[SkillType] = Time.time; // Update the last skill time
            }
            else
            {
                Debug.Log("[Skill][Stone] on cooldown.");
            }
        }

        private void Activate()
        {
            robotPhysic.FreezeMovement();
            robotStats.DisableMove(); // Disable movement
            robotStats.OnColisionEvents += OnCollision; // Subscribe to collision events
            Debug.Log("[Skill][Stone] activated!");

            robotStats.StartCoroutine(DeactivateAfterDuration());

        }

        public IEnumerator DeactivateAfterDuration()
        {
            yield return new WaitForSeconds(Duration);
            Deactivate();
            Debug.Log("[Skill][Stone] deactivated after duration!");
        }

        public void OnCollision(Collision2D collision)
        {
            // Implement the logic for what happens when the robot collides with something while the skill is active
            Debug.Log("Collision detected on " + collision.gameObject.layer + " during stone skill!");

            var physics = collision.gameObject.GetComponent<RobotPhysicController>();

            Vector2 collisionNormal = collision.contacts[0].normal; //[Todo] GetContact()
            Vector2 bounceDirection = Vector2.Reflect(physics.LastVelocity.normalized, collisionNormal);

            float impactForce = physics.LastVelocity.magnitude * BounceBackMultiplier;
            collision.gameObject.GetComponent<RobotPhysicController>().Bounce(bounceDirection, impactForce);
        }

        private void Deactivate()
        {
            // Implement the logic to deactivate the skill
            Debug.Log("[Skill][Stone] deactivated!");
            robotPhysic.ResetFreezeMovement();
            robotStats.OnColisionEvents -= OnCollision; // Unsubscribe from collision events
            robotStats.EnableMove(); // Enable movement

        }
    }
}