using System;
using System.Collections;
using System.Linq;
using SumoBot;
using SumoCore;
using SumoInput;
using SumoManager;
using SumoServices;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace SumoCampaign
{
    /// <summary>
    /// Campaign-only battle configuration. Attach this beside BattleManager to
    /// create a level without changing the reusable battle scene rules.
    /// </summary>
    public class CampaignLevelController : MonoBehaviour
    {
        [Header("Level identity")]
        public string LevelId = "1-1";
        [TextArea] public string Instructions;

        [Header("Player configuration")]
        public InputType PlayerInput = InputType.UI;
        public bool AllowPlayerKeyboard;
        public string PlayerBotId;

        [Header("Opponent configuration")]
        public string OpponentBotId = "AIBot_Primitive";
        [Min(0f)] public float OpponentActionInterval = 1f;

        [Header("Match configuration")]
        public RoundSystem RoundSystem = RoundSystem.BestOf1;
        public float BattleTime = 60f;
        public float CountdownTime = 3f;

        private BattleManager battleManager;
        private BotManager botManager;
        private bool rewardGranted;

        private void Awake()
        {
            battleManager = GetComponent<BattleManager>();
            if (battleManager != null)
            {
                // Do not allow a legacy scene button or other setup callback to
                // start the match while its campaign instructions are visible.
                battleManager.RequireExternalStartConfirmation = true;
            }
        }

        private IEnumerator Start()
        {
            // Wait until every normal Start method (especially BotManager.Start)
            // has created its initial instances, then replace them with the
            // campaign's locked selections.
            yield return null;

            botManager = GetComponent<BotManager>();
            if (battleManager == null || botManager == null)
            {
                Debug.LogError("CampaignLevelController requires BattleManager and BotManager on the same GameObject.", this);
                yield break;
            }

            ApplyConfiguration();
            battleManager.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleChanged);
            ShowPreGamePopup();
        }

        private void OnDisable()
        {
            if (battleManager != null)
                battleManager.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleChanged);
        }

        private void ApplyConfiguration()
        {
            battleManager.LeftInputType = PlayerInput;
            battleManager.RightInputType = InputType.Script;
            battleManager.RoundSystem = RoundSystem;
            battleManager.BattleTime = BattleTime;
            battleManager.CountdownTime = CountdownTime;

            // The source campaign template keeps its bot manager disabled.
            // Campaign levels require it for the scripted opponent (and 3-1's
            // player FSM), while the regular Battle scene remains unchanged.
            botManager.enabled = true;

            // Battle's tick interval remains responsive for the human player.
            // Only the campaign opponent is slowed, which is fair for live input.
            botManager.LeftEnabled = PlayerInput == InputType.Script;
            botManager.RightEnabled = true;
            botManager.LeftActionInterval = 0f;
            botManager.RightActionInterval = OpponentActionInterval;

            // Campaign controls are deliberately explicit: level 1 accepts
            // button actions only, and later levels expose their configured
            // control mode without keyboard fallbacks.
            foreach (InputProvider provider in FindObjectsByType<InputProvider>(FindObjectsInactive.Include, FindObjectsSortMode.None))
                provider.IncludeKeyboard = provider.PlayerSide == PlayerSide.Left && AllowPlayerKeyboard;

            AssignBot(PlayerBotId, PlayerSide.Left, PlayerInput == InputType.Script);
            AssignBot(OpponentBotId, PlayerSide.Right, true);
        }

        private void AssignBot(string botId, PlayerSide side, bool required)
        {
            if (!required)
                return;

            Bot bot = BotUtility.GetAllBotInstances()
                .FirstOrDefault(candidate => string.Equals(candidate.ID, botId, StringComparison.OrdinalIgnoreCase)
                    || string.Equals(candidate.GetType().Name, botId, StringComparison.OrdinalIgnoreCase));

            if (bot == null)
            {
                Debug.LogError($"Campaign level {LevelId} could not find bot '{botId}'.", this);
                return;
            }

            botManager.Assign(bot, side);
        }

        private void ShowPreGamePopup()
        {
            GameObject preBattle = FindSceneObject("Pre-Battle");
            if (preBattle != null)
                preBattle.SetActive(false);

            GameObject ongoingBattle = FindSceneObject("Ongoing Battle");
            if (ongoingBattle != null)
                ongoingBattle.SetActive(true);

            GameObject missionPanel = FindSceneObject("PanelMission");
            if (missionPanel != null)
            {
                missionPanel.SetActive(true);
                return;
            }

            Canvas canvas = FindFirstObjectByType<Canvas>();
            if (canvas == null)
            {
                Debug.LogError("Campaign level needs a Canvas for its pre-game instructions.", this);
                return;
            }

            GameObject overlay = new("Campaign Instructions", typeof(RectTransform), typeof(Image));
            RectTransform overlayRect = overlay.GetComponent<RectTransform>();
            overlayRect.SetParent(canvas.transform, false);
            overlayRect.anchorMin = Vector2.zero;
            overlayRect.anchorMax = Vector2.one;
            overlayRect.offsetMin = overlayRect.offsetMax = Vector2.zero;
            overlay.GetComponent<Image>().color = new Color(0f, 0f, 0f, .78f);

            GameObject panel = new("Panel", typeof(RectTransform), typeof(Image), typeof(VerticalLayoutGroup));
            RectTransform panelRect = panel.GetComponent<RectTransform>();
            panelRect.SetParent(overlay.transform, false);
            panelRect.anchorMin = panelRect.anchorMax = new Vector2(.5f, .5f);
            panelRect.sizeDelta = new Vector2(720f, 390f);
            panel.GetComponent<Image>().color = new Color(.12f, .15f, .2f, 1f);
            VerticalLayoutGroup layout = panel.GetComponent<VerticalLayoutGroup>();
            layout.padding = new RectOffset(36, 36, 32, 32);
            layout.spacing = 18f;
            layout.childAlignment = TextAnchor.MiddleCenter;
            layout.childControlHeight = true;
            layout.childForceExpandHeight = false;

            AddText(panel.transform, $"CAMPAIGN {LevelId}", 38, FontStyles.Bold);
            AddText(panel.transform, Instructions, 24, FontStyles.Normal);

            GameObject start = new("Start Battle", typeof(RectTransform), typeof(Image), typeof(Button), typeof(LayoutElement));
            start.transform.SetParent(panel.transform, false);
            start.GetComponent<Image>().color = new Color(.94f, .67f, .22f, 1f);
            start.GetComponent<LayoutElement>().preferredHeight = 62f;
            AddText(start.transform, "START BATTLE", 26, FontStyles.Bold);
            start.GetComponent<Button>().onClick.AddListener(() =>
            {
                Destroy(overlay);
                battleManager.ConfirmCampaignStart();
            });
        }

        /// <summary>Called by the PanelMission > Tab_Controls Continue button.</summary>
        public void ContinueFromMissionPanel()
        {
            GameObject missionPanel = FindSceneObject("PanelMission");
            if (missionPanel != null)
                missionPanel.SetActive(false);

            battleManager.ConfirmCampaignStart();
        }

        private void OnBattleChanged(EventParameter parameter)
        {
            if (rewardGranted || parameter.BattleState != BattleState.PostBattle_ShowResult)
                return;

            rewardGranted = true;
            int reward = 10;
            if (battleManager.Battle.GetBattleWinner() == BattleWinner.Left)
                reward += 20;

            _ = GrantBattleRewardAsync(reward);
        }

        private async System.Threading.Tasks.Task GrantBattleRewardAsync(int reward)
        {
            if (GameServices.PlayerData == null || GameServices.PlayerData.Current == null)
            {
                Debug.LogWarning($"Campaign {LevelId} reward skipped because no player data is loaded.", this);
                return;
            }

            ServiceResult result = await GameServices.PlayerData.AddCoinsAsync(reward);
            if (!result.Success)
                Debug.LogError($"Campaign {LevelId} could not grant {reward} gold: {result.Error}", this);
        }

        private static GameObject FindSceneObject(string objectName)
        {
            foreach (Transform candidate in FindObjectsByType<Transform>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            {
                if (candidate.name == objectName)
                    return candidate.gameObject;
            }

            return null;
        }

        private static void AddText(Transform parent, string value, float size, FontStyles style)
        {
            GameObject textObject = new("Text", typeof(RectTransform), typeof(TextMeshProUGUI), typeof(LayoutElement));
            textObject.transform.SetParent(parent, false);
            TextMeshProUGUI text = textObject.GetComponent<TextMeshProUGUI>();
            text.text = value;
            text.fontSize = size;
            text.fontStyle = style;
            text.alignment = TextAlignmentOptions.Center;
            text.enableWordWrapping = true;
            text.color = Color.white;
            LayoutElement element = textObject.GetComponent<LayoutElement>();
            element.flexibleHeight = 1f;
        }
    }
}
