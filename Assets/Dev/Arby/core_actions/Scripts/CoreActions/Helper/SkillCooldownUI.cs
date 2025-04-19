using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace RobotCoreAction
{
    /// <summary>
    /// This is just a simple game manager to handle the cooldown of the skills.
    /// It's deletable file
    /// </summary>
    public class SkillCooldownUI : MonoBehaviour
    {
        public static SkillCooldownUI Instance { get; private set; }

        public Dictionary<ERobotSkillType, string> runningSkill = new Dictionary<ERobotSkillType, string>();

        private void Awake()
        {
            // If there is an instance, and it's not me, delete myself.

            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;
            }
        }

        [SerializeField] private TMP_Text textUI;

        public void ShowSkillCooldown(ISkill skill)
        {
            StartCoroutine(SpawnTimer(skill));
        }

        void Update()
        {
            if (runningSkill.Count > 0)
            {
                string text = "";
                foreach (var item in runningSkill)
                {
                    text += item.Value + "\n";
                }
                textUI.text = text;
            }
            else
            {
                textUI.text = "";
            }
        }

        private IEnumerator SpawnTimer(ISkill skill)
        {
            float cd = skill.Cooldown;
            while (cd > 0)
            {
                var s = DisplayTime(cd, skill);
                yield return new WaitForSeconds(1f);
                cd -= 1;
                if (s == 1)
                {
                    runningSkill.Remove(skill.SkillType);
                }
            }
        }

        int DisplayTime(float timeToDisplay, ISkill skill)
        {
            int seconds = Mathf.FloorToInt(timeToDisplay % 60);
            runningSkill[skill.SkillType] = string.Format("Skill {0} is on cooldown in {1}", skill.SkillType.ToString(), seconds);
            return seconds;
        }
    }
}