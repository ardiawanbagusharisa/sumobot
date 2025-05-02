using System.Collections;
using BattleLoop;
using Unity.VisualScripting;
using UnityEngine;

namespace CoreSumoRobot
{

    public abstract class ISkill
    {
        public abstract ERobotSkillType SkillType { get; }
        public abstract float Duration { get; }
        public abstract float Cooldown { get; }
        public virtual void Activate(SumoRobotController controller, SumoRobot robot)
        {
            Debug.Log($"[Skill][{SkillType}] activated!");

            BattleManager.Instance.CurrentRound.SetActionLog(robot.IsLeftSide, $"type=skill;robotId={robot.IdInt};skillType={SkillType};isActive=true");
        }

        public virtual void Deactivate(SumoRobotController controller, SumoRobot robot)
        {
            BattleManager.Instance.CurrentRound.SetActionLog(robot.IsLeftSide, $"type=skill;robotId={robot.IdInt};skillType={SkillType};isActive=false");

            Debug.Log($"[Skill][{SkillType}] deactivated!");
        }
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
        public override ERobotSkillType SkillType => ERobotSkillType.Boost;
        public override float Duration => 5f;
        public override float Cooldown => 10f;
        public float BoostMultiplier => 1.8f;
        #endregion

        private SumoRobot robot;
        private SumoRobotController controller;

        public override void Activate(SumoRobotController controllerParam, SumoRobot robotParam)
        {
            controller = controllerParam;
            robot = robotParam;
            controller.SkillTime.TryGetValue(SkillType, out float lastSkillTime);
            if (BattleManager.Instance.ElapsedTime >= lastSkillTime + Cooldown)
            {
                //Debug
                SkillCooldownUI.Instance.ShowSkillCooldown(this);

                Boost();
                controller.LastRobotSkillType = SkillType;
            }
            else
            {
                Debug.Log("[Skill][Boost] on cooldown.");
            }
        }

        private void Boost()
        {
            base.Activate(controller, robot);
            controller.SetMovementEnabled(true);
            controller.ChangeMoveSpeed(robot.MoveSpeed * BoostMultiplier);
            controller.ChangeDashSpeed(robot.DashSpeed * BoostMultiplier);
            controller.SetBounceResistance(1.1f);
            controller.StartCoroutine(DeactivateAfterDuration());
        }

        public IEnumerator DeactivateAfterDuration()
        {
            yield return new WaitForSeconds(Duration);
            Deactivate(controller, robot);
        }

        public override void Deactivate(SumoRobotController controller, SumoRobot robot)
        {
            controller.ResetMoveSpeed();
            controller.ResetDashSpeed();
            controller.ResetBounceResistance();
            controller.LastRobotSkillType = ERobotSkillType.None;

            base.Deactivate(controller, robot);
        }
    }

    public class StoneSkill : ISkill
    {
        #region Stone Skill Stat
        public override ERobotSkillType SkillType => ERobotSkillType.Stone;
        public override float Duration => 5f;
        public override float Cooldown => 10f;
        public float BounceBackMultiplier => 10.0f;
        #endregion

        private SumoRobot robot;
        private SumoRobotController controller;

        public override void Activate(SumoRobotController controllerParam, SumoRobot robotParam)
        {
            robot = robotParam;
            controller = controllerParam;
            controller.SkillTime.TryGetValue(SkillType, out float lastSkillTime);
            if (BattleManager.Instance.ElapsedTime >= lastSkillTime + Cooldown)
            {
                //Debug
                SkillCooldownUI.Instance.ShowSkillCooldown(this);

                Stone();
                controller.LastRobotSkillType = SkillType;
            }
            else
            {
                Debug.Log("[Skill][Stone] on cooldown.");
            }
        }

        private void Stone()
        {
            base.Activate(controller, robot);

            controller.FreezeMovement();
            controller.SetMovementEnabled(false);
            controller.SetBounceResistance(BounceBackMultiplier);
            controller.StartCoroutine(DeactivateAfterDuration());

        }

        public IEnumerator DeactivateAfterDuration()
        {
            yield return new WaitForSeconds(Duration);
            Deactivate(controller, robot);
        }

        public override void Deactivate(SumoRobotController controller, SumoRobot robot)
        {
            controller.ResetFreezeMovement();
            controller.ResetBounceResistance();
            controller.SetMovementEnabled(true);
            controller.LastRobotSkillType = ERobotSkillType.None;

            base.Deactivate(controller, robot);
        }

    }

}