using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

namespace CoreSumoRobot
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

        public void ShowSkillCooldown(SumoSkill skill, ERobotSkillType type)
        {
            StartCoroutine(SpawnTimer(skill, type));
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

        private IEnumerator SpawnTimer(SumoSkill skill, ERobotSkillType type)
        {
            float cd = skill.GetCooldownInfo(type);

            while (cd > 0)
            {
                var s = DisplayTime(cd, type);
                yield return new WaitForSeconds(1f);
                cd -= 1;
                if (s < 1)
                {
                    runningSkill.Remove(skill.CurrentSkillType);
                }
            }
        }

        int DisplayTime(float timeToDisplay, ERobotSkillType type)
        {
            int seconds = Mathf.FloorToInt(timeToDisplay % 60);
            runningSkill[type] = $"Skill {type} is on cooldown in {seconds}"; ;
            return seconds;
        }
    }
}