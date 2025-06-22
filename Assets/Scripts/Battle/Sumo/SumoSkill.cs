using System;
using System.Collections;
using UnityEngine;

namespace CoreSumo
{
    public enum SkillType
    {
        Boost = 0,
        Stone = 1,
    }

    [Serializable]
    public class SumoSkill
    {
        #region General info properties
        public SkillType Type = SkillType.Boost;
        public bool IsActive = false;
        #endregion

        #region Skill Stat properties
        public float StoneCooldown = 10f;
        public float StoneDuration = 5f;
        public float StoneBounceBackMultiplier = 10f;
        public float BoostCooldown = 10f;
        public float BoostDuration = 5f;
        public float BoostMultiplier = 1.8f;
        #endregion

        #region Runtime (readonly) properties
        private float stoneLastTimeUsed;
        private float boostLastTimeUsed;
        private SumoController controller;
        #endregion

        public SumoSkill(SumoController controller)
        {
            this.controller = controller;
        }

        #region Runtime properties 
        public float CooldownAmount()
        {
            float lastUsedSkill = 0;

            switch (Type)
            {
                case SkillType.Boost:
                    lastUsedSkill = boostLastTimeUsed;
                    break;
                case SkillType.Stone:
                    lastUsedSkill = stoneLastTimeUsed;
                    break;
            }

            float skillCooldown = Type == SkillType.Boost ? BoostCooldown : StoneCooldown;
            float cooldownAmount = lastUsedSkill + skillCooldown - BattleManager.Instance.ElapsedTime;
            return cooldownAmount;
        }

        public float CooldownAmountNormalized => 1 - (CooldownAmount() / SkillCooldown);
        public bool IsSkillCooldown => CooldownAmount() >= 0f;
        public float SkillDuration => Type == SkillType.Boost ? BoostDuration : StoneDuration;
        public float SkillCooldown => Type == SkillType.Boost ? BoostCooldown : StoneCooldown;
        #endregion

        #region Activation and cooldown methods
        public void Reset()
        {
            boostLastTimeUsed = 0;
            stoneLastTimeUsed = 0;
            IsActive = false;
        }

        public bool Activate(ISumoAction action)
        {
            Debug.Log($"[Skill][{Type}] activated!");
            IsActive = true;
            controller.Log(action);
            switch (Type)
            {
                case SkillType.Boost:
                    ActivateBoost();
                    break;
                case SkillType.Stone:
                    ActivateStone();
                    break;
            }

            controller.StartCoroutine(OnAfterDuration(Type));
            controller.StartCoroutine(OnAfterCooldown(Type));
            return true;
        }

        public void ActivateBoost()
        {
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

        private IEnumerator OnAfterDuration(SkillType type)
        {
            float duration = type == SkillType.Boost ? BoostDuration : StoneDuration;
            yield return new WaitForSeconds(duration);
            IsActive = false;
            switch (type)
            {
                case SkillType.Boost:
                    controller.ResetMoveSpeed();
                    controller.ResetDashSpeed();
                    break;
                case SkillType.Stone:
                    controller.ResetFreezeMovement();
                    controller.ResetBounceResistance();
                    controller.SetMovementEnabled(true);
                    break;
            }
        }

        private IEnumerator OnAfterCooldown(SkillType type)
        {
            float cooldown = type == SkillType.Boost ? BoostCooldown : StoneCooldown;
            yield return new WaitForSeconds(cooldown);

            Debug.Log($"[Skill][{Type}] cooldown end!");
        }
        #endregion
    }
}