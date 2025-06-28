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
        public float TotalCooldown = 10f;
        public float TotalDuration = 5f;
        public float StoneMultiplier = 10f;
        public float BoostMultiplier = 1.8f;
        #endregion

        #region Runtime (readonly) properties
        private float usedAt;
        private SumoController controller;
        #endregion

        public SumoSkill(SumoController controller)
        {
            this.controller = controller;
        }

        static public SumoSkill CreateSkill(
            SumoController controller,
            SkillType type,
            float cooldown = 10f,
            float duration = 5f
            )
        {
            SumoSkill skill = new(controller)
            {
                Type = type,
                TotalCooldown = cooldown,
                TotalDuration = duration,
            };
            return skill;
        }

        #region Runtime properties 
        public float Cooldown => usedAt + TotalCooldown - BattleManager.Instance.ElapsedTime;
        public float CooldownNormalized => 1 - (Cooldown / TotalCooldown);
        public bool IsSkillOnCooldown => Cooldown >= 0f;
        #endregion

        #region Activation and cooldown methods
        public void Reset()
        {
            usedAt = 0;
            IsActive = false;
        }

        public bool Activate(ISumoAction action)
        {
            action.Type = (Type == SkillType.Boost) ? ActionType.SkillBoost : ActionType.SkillStone;
            if (IsSkillOnCooldown)
            {
                Debug.Log($"[Skill][{Type}] is on cooldown");
                return false;
            }

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

            controller.StartCoroutine(OnAfterDuration());
            controller.StartCoroutine(OnAfterCooldown());
            return true;
        }

        public void ActivateBoost()
        {
            usedAt = BattleManager.Instance.ElapsedTime;
            controller.MoveSpeed *= BoostMultiplier;
            controller.DashSpeed *= BoostMultiplier;
        }

        public void ActivateStone()
        {
            usedAt = BattleManager.Instance.ElapsedTime;
            controller.FreezeMovement();
            controller.BounceResistance *= StoneMultiplier;
        }

        private IEnumerator OnAfterDuration()
        {
            yield return new WaitForSeconds(TotalDuration);
            IsActive = false;
            switch (Type)
            {
                case SkillType.Boost:
                    controller.ResetMoveSpeed();
                    controller.ResetDashSpeed();
                    break;
                case SkillType.Stone:
                    controller.ResetFreezeMovement();
                    controller.ResetBounceResistance();
                    break;
            }
        }

        private IEnumerator OnAfterCooldown()
        {
            yield return new WaitForSeconds(TotalCooldown);

            Debug.Log($"[Skill][{Type}] cooldown end!");
        }
        #endregion
    }
}