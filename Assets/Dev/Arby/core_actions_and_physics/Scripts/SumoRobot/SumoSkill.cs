using System.Collections;
using BattleLoop;
using UnityEngine;

namespace CoreSumoRobot
{
    public enum ERobotSkillType
    {
        Stone,
        Boost,
        None,
    }

    public class SumoSkill
    {
        public ERobotSkillType CurrentSkillType;

        #region Stone Stat
        public float StoneCooldown = 10f;
        public float StoneDuration = 5f;
        public float StoneBounceBackMultiplier = 10f;
        #endregion

        #region Boost Stat
        public float BoostCooldown = 10f;
        public float BoostDuration = 5f;
        public float BoostMultiplier = 1.8f;
        #endregion

        private float stoneLastTimeUsed;
        private float boostLastTimeUsed;

        private SumoRobotController controller;
        private SumoRobot sumo;

        // This function provides flexibility of knowing the usage of skill cooldown
        // call SumoSkill.IsSkillCooldown() -> returns the skill cooldown of current skill
        // call SumoSkill.IsSkillCooldown(ERobotSkillType.Boost) -> returns skill cooldown only for the boost
        public bool IsSkillCooldown(ERobotSkillType type = ERobotSkillType.None)
        {
            float lastUsedSkill = 0;

            switch (type)
            {
                case ERobotSkillType.Boost:
                    lastUsedSkill = boostLastTimeUsed;
                    break;
                case ERobotSkillType.Stone:
                    lastUsedSkill = stoneLastTimeUsed;
                    break;
                case ERobotSkillType.None:
                    if (CurrentSkillType == ERobotSkillType.None)
                    {
                        return false;
                    }

                    lastUsedSkill = CurrentSkillType == ERobotSkillType.Boost ? boostLastTimeUsed : stoneLastTimeUsed;
                    break;
            }

            float skillCooldown = type == ERobotSkillType.Boost ? BoostCooldown : StoneCooldown;

            Debug.Log($"IsSkillCooldown: elapsedTime={BattleManager.Instance.ElapsedTime}, lastUsedSkill={lastUsedSkill}, skillCooldown={skillCooldown}");
            if (BattleManager.Instance.ElapsedTime >= lastUsedSkill + skillCooldown)
            {
                return false;
            }
            return true;
        }

        public float GetCooldownInfo(ERobotSkillType type = ERobotSkillType.None)
        {
            switch (type)
            {
                case ERobotSkillType.Boost:
                    return BoostCooldown;
                case ERobotSkillType.Stone:
                    return StoneCooldown;
                case ERobotSkillType.None:
                    if (CurrentSkillType == ERobotSkillType.None)
                    {
                        return -1;
                    }
                    return CurrentSkillType == ERobotSkillType.Boost ? BoostCooldown : StoneCooldown;
            }
            return -1;
        }

        public void Reset()
        {
            boostLastTimeUsed = 0;
            stoneLastTimeUsed = 0;
            CurrentSkillType = ERobotSkillType.None;
        }

        public void Activate(SumoRobotController controllerParam, SumoRobot sumoParam, ERobotSkillType skillTypeParam)
        {
            sumo = sumoParam;
            controller = controllerParam;
            CurrentSkillType = skillTypeParam;

            if (IsSkillCooldown() == false)
            {
                Debug.Log($"[Skill][{CurrentSkillType}] activated!");
                BattleManager.Instance.CurrentRound.SetActionLog(sumoParam.IsLeftSide, $"type=skill;robotId={sumoParam.IdInt};skillType={CurrentSkillType};isActive=true");

                switch (CurrentSkillType)
                {
                    case ERobotSkillType.Boost:
                        ActivateBoost();
                        break;
                    case ERobotSkillType.Stone:
                        ActivateStone();
                        break;
                }

                controller.StartCoroutine(OnAfterDuration(CurrentSkillType));
                controller.StartCoroutine(OnAfterCooldown(CurrentSkillType));
            }
            else
            {
                Debug.Log($"[Skill][{CurrentSkillType}] is on cooldown!");
            }

        }

        public void ActivateBoost()
        {
            SkillCooldownUI.Instance.ShowSkillCooldown(this, ERobotSkillType.Boost);
            boostLastTimeUsed = BattleManager.Instance.ElapsedTime;

            controller.SetMovementEnabled(true);
            controller.ChangeMoveSpeed(sumo.MoveSpeed * BoostMultiplier);
            controller.ChangeDashSpeed(sumo.DashSpeed * BoostMultiplier);

        }

        public void ActivateStone()
        {
            SkillCooldownUI.Instance.ShowSkillCooldown(this, ERobotSkillType.Stone);
            stoneLastTimeUsed = BattleManager.Instance.ElapsedTime;

            controller.FreezeMovement();
            controller.SetMovementEnabled(false);
            controller.SetBounceResistance(StoneBounceBackMultiplier);
        }

        private IEnumerator OnAfterDuration(ERobotSkillType type)
        {
            float duration = type == ERobotSkillType.Boost ? BoostDuration : StoneDuration;
            yield return new WaitForSeconds(duration);

            switch (type)
            {
                case ERobotSkillType.Boost:
                    controller.ResetMoveSpeed();
                    controller.ResetDashSpeed();
                    break;
                case ERobotSkillType.Stone:
                    controller.ResetFreezeMovement();
                    controller.ResetBounceResistance();
                    controller.SetMovementEnabled(true);
                    break;
            }

            BattleManager.Instance.CurrentRound.SetActionLog(sumo.IsLeftSide, $"type=skill;robotId={sumo.IdInt};skillType={CurrentSkillType};isActive=false");
        }

        private IEnumerator OnAfterCooldown(ERobotSkillType type)
        {
            float cooldown = type == ERobotSkillType.Boost ? BoostCooldown : StoneCooldown;
            yield return new WaitForSeconds(cooldown);

            CurrentSkillType = ERobotSkillType.None;
            Debug.Log($"[Skill][{CurrentSkillType}] cooldown end!");
        }


    }


}