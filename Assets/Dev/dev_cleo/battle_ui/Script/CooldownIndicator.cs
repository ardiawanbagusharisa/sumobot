using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Events;

public class CooldownManager : MonoBehaviour
{
    [System.Serializable]
    public class CooldownUI
    {
        public Button button;
        public Image cooldownImage;
        public float cooldownTime = 5f;
        public AudioClip cooldownSound;
        public Color activeColor = Color.white;
        public Color cooldownColor = new Color(1, 1, 1, 0.5f);
    }

    public CooldownUI skillBoost;
    public CooldownUI stone;

    private AudioSource audioSource;

    private float timerSkill = 0f;
    private float timerStone = 0f;

    private void Start()
    {
        audioSource = gameObject.AddComponent<AudioSource>();
        ResetCooldownUI(skillBoost);
        ResetCooldownUI(stone);
    }

    private void Update()
    {
        UpdateCooldown(ref timerSkill, skillBoost);
        UpdateCooldown(ref timerStone, stone);
    }

    public void TriggerSkillBoost()
    {
        if (timerSkill <= 0f)
        {
            timerSkill = skillBoost.cooldownTime;
            ActivateCooldown(skillBoost);
        }
    }

    public void TriggerStone()
    {
        if (timerStone <= 0f)
        {
            timerStone = stone.cooldownTime;
            ActivateCooldown(stone);
        }
    }

    void UpdateCooldown(ref float timer, CooldownUI ui)
    {
        if (timer > 0f)
        {
            timer -= Time.deltaTime;
            float fill = Mathf.Clamp01(timer / ui.cooldownTime);
            ui.cooldownImage.fillAmount = fill;

            if (timer <= 0f)
            {
                ResetCooldownUI(ui);
            }
        }
    }

    void ActivateCooldown(CooldownUI ui)
    {
        if (ui.button) ui.button.interactable = false;
        if (ui.cooldownImage)
        {
            ui.cooldownImage.fillAmount = 1f;
            ui.cooldownImage.color = ui.cooldownColor;
        }

        if (ui.cooldownSound)
        {
            audioSource.PlayOneShot(ui.cooldownSound);
        }
    }

    void ResetCooldownUI(CooldownUI ui)
    {
        if (ui.button) ui.button.interactable = true;
        if (ui.cooldownImage)
        {
            ui.cooldownImage.fillAmount = 0f;
            ui.cooldownImage.color = ui.activeColor;
        }
    }
}
