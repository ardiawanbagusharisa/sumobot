using System;
using System.Collections;
using BattleLoop;
using UnityEngine;

namespace CoreSumoRobot
{
    public enum ERobotSkillType
    {
        Boost = 0,
        Stone = 1,
        None = 2,
    }


    [Serializable]
    public class SumoSkill
    {
        public ERobotSkillType Type = ERobotSkillType.Boost;
        public bool IsActive = false;

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

        public SumoSkill(SumoRobotController controller)
        {
            this.controller = controller;
        }

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
                    if (Type == ERobotSkillType.None)
                    {
                        return false;
                    }

                    lastUsedSkill = Type == ERobotSkillType.Boost ? boostLastTimeUsed : stoneLastTimeUsed;
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

        public void Reset()
        {
            boostLastTimeUsed = 0;
            stoneLastTimeUsed = 0;
            IsActive = false;
        }

        public bool Activate(ERobotSkillType skillTypeParam)
        {
            Type = skillTypeParam;

            if (IsSkillCooldown() == false)
            {
                Debug.Log($"[Skill][{Type}] activated!");

                IsActive = true;
                switch (Type)
                {
                    case ERobotSkillType.Boost:
                        ActivateBoost();
                        break;
                    case ERobotSkillType.Stone:
                        ActivateStone();
                        break;
                }

                controller.StartCoroutine(OnAfterDuration(Type));
                controller.StartCoroutine(OnAfterCooldown(Type));
                return true;
            }
            else
            {
                Debug.Log($"[Skill][{Type}] is on cooldown!");
            }

            return false;

        }

        public void ActivateBoost()
        {
            // BattleUIManager.Instance.ShowSkillCooldown(this, ERobotSkillType.Boost);
            boostLastTimeUsed = BattleManager.Instance.ElapsedTime;
            controller.SetMovementEnabled(true);
            controller.MoveSpeed *= BoostMultiplier;
            controller.DashSpeed *= BoostMultiplier;
        }

        public void ActivateStone()
        {
            stoneLastTimeUsed = BattleManager.Instance.ElapsedTime;
            controller.FreezeMovement();
            controller.SetMovementEnabled(false);
            controller.BounceResistance *= StoneBounceBackMultiplier;
        }

        private IEnumerator OnAfterDuration(ERobotSkillType type)
        {
            float duration = type == ERobotSkillType.Boost ? BoostDuration : StoneDuration;
            yield return new WaitForSeconds(duration);

            IsActive = false;
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
        }

        private IEnumerator OnAfterCooldown(ERobotSkillType type)
        {
            float cooldown = type == ERobotSkillType.Boost ? BoostCooldown : StoneCooldown;
            yield return new WaitForSeconds(cooldown);

            Debug.Log($"[Skill][{Type}] cooldown end!");
        }


    }


}