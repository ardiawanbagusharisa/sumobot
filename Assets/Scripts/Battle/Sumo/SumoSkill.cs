using System;
using System.Collections;
using SumoManager;
using UnityEngine;

namespace SumoCore
{
    public enum SkillType
    {
        Boost = 0,
        Stone = 1,
    }

    public static class SkillTypeExtension
    {
        public static ActionType ToActionType(this SkillType type)
        {
            if (type == SkillType.Boost)
            {
                return ActionType.SkillBoost;
            }
            else
            {
                return ActionType.SkillStone;
            }
        }
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
        public float StoneMultiplier = 7f;
        public float BoostMultiplier = 1.8f;
        #endregion

        #region Runtime (readonly) properties
        public bool IsInitialized = false;
        private float usedAt;
        private readonly SumoController controller;
        #endregion

        public SumoSkill(SumoController controller)
        {
            this.controller = controller;
        }

        static public SumoSkill CreateSkill(
            SumoController controller,
            SkillType type,
            float cooldown = 10f,
            float duration = 5f)
        {
            SumoSkill skill = new(controller)
            {
                Type = type,
                TotalCooldown = cooldown,
                TotalDuration = duration,
                IsInitialized = true,
            };
            return skill;
        }

        #region Runtime properties 
        public float Cooldown => usedAt + TotalCooldown - BattleManager.Instance.ElapsedTime;
        public float CooldownNormalized => 1 - (Cooldown / TotalCooldown);
        public bool IsSkillOnCooldown => Cooldown >= 0f;
        private Coroutine DurationRoutine;
        private Coroutine CooldownRoutine;
        #endregion

        #region Activation and cooldown methods
        public void Reset()
        {
            usedAt = 0;
            IsActive = false;

            if (DurationRoutine != null)
            {
                controller.StopCoroutine(DurationRoutine);
                DurationRoutine = null;
            }
            if (CooldownRoutine != null)
            {
                controller.StopCoroutine(CooldownRoutine);
                CooldownRoutine = null;
            }

            controller.RigidBody.constraints = RigidbodyConstraints2D.None;
            controller.ResetBounceResistance();
            controller.ResetMoveSpeed();
            controller.ResetDashSpeed();
        }

        public bool Activate(ISumoAction action)
        {
            if (controller.IsMovementDisabled)
                return false;

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

            DurationRoutine = controller.StartCoroutine(OnAfterDuration());
            CooldownRoutine = controller.StartCoroutine(OnAfterCooldown());
            return true;
        }

        public void ActivateBoost()
        {
            SFXManager.Instance.Play2D("actions_boost");
            usedAt = BattleManager.Instance.ElapsedTime;
            controller.MoveSpeed *= BoostMultiplier;
            controller.DashSpeed *= BoostMultiplier;
        }

        public void ActivateStone()
        {
            SFXManager.Instance.Play2D("actions_stone");
            usedAt = BattleManager.Instance.ElapsedTime;
            controller.RigidBody.constraints = RigidbodyConstraints2D.FreezePosition;
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
                    controller.RigidBody.constraints = RigidbodyConstraints2D.None;
                    controller.ResetBounceResistance();
                    break;
            }
        }

        private IEnumerator OnAfterCooldown()
        {
            yield return new WaitForSeconds(TotalCooldown);

            Debug.Log($"[Skill][{Type}] cooldown end!");
        }

        public override string ToString()
        {
            string typeLabel = Type.ToString().ToUpper();
            string cooldownStatus = IsSkillOnCooldown ? $"{Cooldown:F1}s ({CooldownNormalized:P0})" : "Ready";
            string activeStatus = IsActive ? "Active" : "Inactive";

            return $"[Skill: {typeLabel}]\n" +
                   $"- Status     : {activeStatus}\n" +
                   $"- Cooldown   : {cooldownStatus}\n" +
                   $"- Duration   : {TotalDuration:F1}s\n" +
                   $"- Multiplier : {(Type == SkillType.Boost ? BoostMultiplier : StoneMultiplier):F1}";
        }
        #endregion
    }
}