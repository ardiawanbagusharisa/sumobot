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

        public float SkillCooldown()
        {
            float lastUsedSkill = 0;

            switch (Type)
            {
                case ERobotSkillType.Boost:
                    lastUsedSkill = boostLastTimeUsed;
                    break;
                case ERobotSkillType.Stone:
                    lastUsedSkill = stoneLastTimeUsed;
                    break;
            }

            float skillCooldown = Type == ERobotSkillType.Boost ? BoostCooldown : StoneCooldown;

            float cooldownAmount = lastUsedSkill + skillCooldown - BattleManager.Instance.ElapsedTime;
            return cooldownAmount;
        }

        public bool IsSkillCooldown => SkillCooldown() >= 0f;
        public float SkillDuration => Type == ERobotSkillType.Boost ? BoostDuration : StoneDuration;

        public void Reset()
        {
            boostLastTimeUsed = 0;
            stoneLastTimeUsed = 0;
            IsActive = false;
        }

        public bool Activate(ERobotSkillType skillTypeParam)
        {
            Type = skillTypeParam;

            // Check whether the skill is ready or not
            if (!IsSkillCooldown)
            {
                Debug.Log($"[Skill][{Type}] activated!");
                controller.ActionLoggers["Skill"].Call(Type.ToString());
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
            // LogManager.LogRoundEvent(
            //         actor: controller.Side == PlayerSide.Left ? LogActorType.LeftPlayer : LogActorType.RightPlayer,
            //         action: LogActionType.Player_Skill,
            //         detail: $"type={Type};active=false");
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