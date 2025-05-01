using System.Collections;
using BattleLoop;
using UnityEngine;

namespace CoreSumoRobot
{

    public interface ISkill
    {
        ERobotSkillType SkillType { get; }
        float Duration { get; }
        float Cooldown { get; }
        void Execute(SumoRobotController controller, SumoRobot robot);
    }
    public enum ERobotSkillType
    {
        Stone,
        Boost,
        None,
    }

    public class BoostSkill : ISkill
    {
        #region Boost Skill Stat
        public ERobotSkillType SkillType => ERobotSkillType.Boost;
        public float Duration => 5f;
        public float Cooldown => 10f;
        public float BoostMultiplier => 1.8f;
        #endregion

        private SumoRobot robot;
        private SumoRobotController controller;


        public void Execute(SumoRobotController controllerParam, SumoRobot robotParam)
        {
            OnExecute(controllerParam, robotParam);
        }

        private void OnExecute(SumoRobotController controllerParam, SumoRobot robotParam)
        {
            controller = controllerParam;
            robot = robotParam;
            controller.SkillTime.TryGetValue(SkillType, out float lastSkillTime);
            if (BattleManager.Instance.ElapsedTime >= lastSkillTime + Cooldown)
            {
                //Debug
                SkillCooldownUI.Instance.ShowSkillCooldown(this);

                Activate();

                controller.LastRobotSkillType = SkillType;
            }
            else
            {
                Debug.Log("[Skill][Boost] on cooldown.");
            }
        }

        private void Activate()
        {
            Debug.Log("[Skill][Boost] activated!");
            controller.SetMovementEnabled(true);
            controller.ChangeMoveSpeed(robot.MoveSpeed * BoostMultiplier);
            controller.ChangeDashSpeed(robot.DashSpeed * BoostMultiplier);
            controller.SetBounceResistance(1.1f);
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
            controller.ResetBounceResistance();
            controller.LastRobotSkillType = ERobotSkillType.None;
            Debug.Log("[Skill][Boost] deactivated!");
        }
    }

    public class StoneSkill : ISkill
    {
        #region Stone Skill Stat
        public ERobotSkillType SkillType => ERobotSkillType.Stone;
        public float Duration => 5f;
        public float Cooldown => 10f;
        public float BounceBackMultiplier => 10.0f;
        #endregion

        private SumoRobot robot;
        private SumoRobotController controller;



        public void Execute(SumoRobotController controllerParam, SumoRobot robotParam)
        {
            robot = robotParam;
            controller = controllerParam;
            controller.SkillTime.TryGetValue(SkillType, out float lastSkillTime);
            if (BattleManager.Instance.ElapsedTime >= lastSkillTime + Cooldown)
            {
                //Debug
                SkillCooldownUI.Instance.ShowSkillCooldown(this);

                Activate();
                controller.LastRobotSkillType = SkillType;
            }
            else
            {
                Debug.Log("[Skill][Stone] on cooldown.");
            }
        }

        private void Activate()
        {
            controller.FreezeMovement();
            controller.SetMovementEnabled(false);
            controller.SetBounceResistance(BounceBackMultiplier);
            Debug.Log("[Skill][Stone] activated!");

            controller.StartCoroutine(DeactivateAfterDuration());

        }

        public IEnumerator DeactivateAfterDuration()
        {
            yield return new WaitForSeconds(Duration);
            Deactivate();
        }

        private void Deactivate()
        {
            // Implement the logic to deactivate the skill
            Debug.Log("[Skill][Stone] deactivated!");
            controller.ResetFreezeMovement();
            controller.ResetBounceResistance();
            controller.SetMovementEnabled(true);
            controller.LastRobotSkillType = ERobotSkillType.None;
        }

    }

}