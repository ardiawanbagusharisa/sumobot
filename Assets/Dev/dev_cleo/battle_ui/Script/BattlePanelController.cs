using UnityEngine;
using UnityEngine.UI;
using TMPro; // Jika pakai TextMeshPro
using System.Collections;

public class BattlePanelController : MonoBehaviour
{
    public GameObject preBattlePanel;
    public GameObject battlePanel;
    public GameObject postBattlePanel;

    public Button startBattleButton;
    public Button finishBattleButton;
    public Button outBattleButton;

    public float fadeDuration = 0.5f;

    public TextMeshProUGUI timerText; // Drag dari Inspector
    // public Text timerText; // Kalau pakai UI.Text biasa
    public SupporterSpawner supporterSpawner;

    private float battleTime = 0f;
    private bool isBattleRunning = false;

    private void Start()
    {
        InitCanvasGroup(preBattlePanel, true);
        InitCanvasGroup(battlePanel, false);
        InitCanvasGroup(postBattlePanel, false);

        if (startBattleButton != null)
            startBattleButton.onClick.AddListener(() =>
            {
                battleTime = 0f;
                isBattleRunning = true;
                StartCoroutine(SwitchPanel(preBattlePanel, battlePanel));
            });

        if (finishBattleButton != null)
            finishBattleButton.onClick.AddListener(() =>
            {
                isBattleRunning = false;
                StartCoroutine(SwitchPanel(battlePanel, postBattlePanel));
            });

        if (outBattleButton != null)
            outBattleButton.onClick.AddListener(() =>
            {
                isBattleRunning = false;
                StartCoroutine(SwitchPanel(battlePanel, preBattlePanel));
            });
    }

    private void Update()
    {
        if (isBattleRunning)
        {
            battleTime += Time.deltaTime;

            int minutes = Mathf.FloorToInt(battleTime / 60f);
            int seconds = Mathf.FloorToInt(battleTime % 60f);

            if (timerText != null)
                timerText.text = string.Format("{0:00}:{1:00}", minutes, seconds);
        }
    }

    void InitCanvasGroup(GameObject panel, bool active)
    {
        var cg = panel.GetComponent<CanvasGroup>();
        if (cg == null)
            cg = panel.AddComponent<CanvasGroup>();

        cg.alpha = active ? 1f : 0f;
        cg.interactable = active;
        cg.blocksRaycasts = active;
        panel.SetActive(true); // semua panel aktif, tapi yang tidak dipakai transparan dan non-interaktif
    }

    IEnumerator SwitchPanel(GameObject fromPanel, GameObject toPanel)
    {
        CanvasGroup fromCG = fromPanel.GetComponent<CanvasGroup>();
        CanvasGroup toCG = toPanel.GetComponent<CanvasGroup>();

        if (toCG == null)
            toCG = toPanel.AddComponent<CanvasGroup>();
        if (fromCG == null)
            fromCG = fromPanel.AddComponent<CanvasGroup>();

        toPanel.SetActive(true);
        toCG.alpha = 0f;
        toCG.interactable = false;
        toCG.blocksRaycasts = false;

        float time = 0f;

        while (time < fadeDuration)
        {
            float t = time / fadeDuration;
            if (fromCG != null) fromCG.alpha = 1f - t;
            if (toCG != null) toCG.alpha = t;
            time += Time.deltaTime;
            yield return null;
        }

        if (fromCG != null)
        {
            fromCG.alpha = 0f;
            fromCG.interactable = false;
            fromCG.blocksRaycasts = false;
        }

        if (toCG != null)
        {
            toCG.alpha = 1f;
            toCG.interactable = true;
            toCG.blocksRaycasts = true;
        }

        fromPanel.SetActive(false);

        if (startBattleButton != null)
            startBattleButton.onClick.AddListener(() =>
            {
                battleTime = 0f;
                isBattleRunning = true;
                supporterSpawner?.SpawnSupporters();
                StartCoroutine(SwitchPanel(preBattlePanel, battlePanel));
            });
    }
}
