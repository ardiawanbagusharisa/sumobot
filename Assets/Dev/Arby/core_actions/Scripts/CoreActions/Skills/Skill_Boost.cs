
using System.Collections;
using UnityEngine;

namespace RobotCoreAction
{
    public class BoostSkill : MonoBehaviour, ISkill
    {
        private RobotActionController robotActionController;
        private RobotStats robotStats;
        private RobotPhysicController robotPhysic;

        public float Duration => 5f;
        public float Cooldown => 10f;

        public float BoostMultiplier => 1.8f;

        public ERobotSkillType SkillType => ERobotSkillType.Boost;

        public void Execute(RobotActionController controller, RobotStats stats, RobotPhysicController physic)
        {

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
                Debug.Log("[Skill][Boost] on cooldown.");
            }
        }

        private void Activate()
        {
            Debug.Log("[Skill][Boost] activated!");
            robotStats.EnableMove(); // Enable movement
            robotStats.ChangeMoveSpeed(robotStats.MoveSpeed * BoostMultiplier);
            robotStats.ChangeDashSpeed(robotStats.DashSpeed * BoostMultiplier);

            robotStats.StartCoroutine(DeactivateAfterDuration());
        }

        public IEnumerator DeactivateAfterDuration()
        {
            yield return new WaitForSeconds(Duration);
            Deactivate();
            Debug.Log("[Skill][Boost] deactivated after duration!");
        }

        private void Deactivate()
        {
            robotStats.ResetMoveSpeed();
            robotStats.ResetDashSpeed();
            Debug.Log("[Skill][Boost] deactivated!");
        }
    }
}