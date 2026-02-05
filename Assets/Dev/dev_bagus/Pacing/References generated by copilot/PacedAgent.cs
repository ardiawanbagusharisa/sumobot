using System;
using System.Collections.Generic;
using System.Linq;
using SumoCore;
using SumoInput;
using SumoLog;
using SumoManager;
using UnityEngine;

namespace SumoBot
{
    /// <summary>
    /// Generic pacing wrapper that applies pacing control to any bot.
    /// Automatically loads the corresponding pacing profile based on the wrapped bot type.
    /// Intercepts actions, logs them, and filters based on pacing thresholds.
    /// </summary>
    [CreateAssetMenu(menuName = "SumoBot/PacedAgent", fileName = "PacedAgent")]
    public class PacedAgent : Bot
    {
        [SerializeField]
        private Bot wrappedBot;

        [Header("Pacing")]
        [SerializeField]
        private PacingProfile pacingProfile;

        [Range(0f, 1f)]
        public float pacingAggression = 0.35f;

        private SumoAPI api;
        private PacingController pacingController;
        private PacingFrame currentPacingFrame;
        private readonly ActionHistory actionHistory = new();
        private bool isInitialized = false;

        public override string ID => wrappedBot?.ID ?? "PacedAgent_Unknown";
        public override SkillType DefaultSkillType => wrappedBot?.DefaultSkillType ?? SkillType.Boost;
        public override bool UseAsync => wrappedBot?.UseAsync ?? false;

        public override void OnBotInit(SumoAPI botAPI)
        {
            api = botAPI;
            actionHistory.Reset();

            // Auto-load pacing profile if not assigned
            if (pacingProfile == null)
            {
                pacingProfile = LoadPacingProfile();
            }

            if (pacingProfile != null)
            {
                pacingController = new PacingController(pacingProfile);
                pacingController.Init(api);
                Debug.Log($"[{ID}] Pacing initialized with profile: {pacingProfile.name}");
            }
            else
            {
                Debug.Log($"[{ID}] No pacing profile loaded");
            }

            if (wrappedBot != null)
            {
                wrappedBot.OnBotInit(botAPI);
                isInitialized = true;
            }
        }

        public override void OnBotUpdate()
        {
            if (!isInitialized || wrappedBot == null)
                return;

            // Update pacing each frame
            if (pacingController != null)
            {
                currentPacingFrame = pacingController.Tick();
            }

            // Clear any previous commands
            ClearCommands();

            // Call wrapped bot's update (which will queue actions)
            wrappedBot.OnBotUpdate();

            // Intercept and log the queued actions before submitting
            if (pacingController != null)
            {
                InterceptAndFilterActions();
            }

            // Submit the (possibly filtered) actions
            Submit();
        }

        public override void OnBotCollision(BounceEvent bounceEvent)
        {
            pacingController?.RegisterCollision(bounceEvent);
            wrappedBot?.OnBotCollision(bounceEvent);
        }

        public override void OnBattleStateChanged(BattleState state, BattleWinner? winner)
        {
            wrappedBot?.OnBattleStateChanged(state, winner);

            if (state == BattleState.Battle_End)
            {
                LogActionSummary();
            }
        }

        public override void OnBotDestroy()
        {
            wrappedBot?.OnBotDestroy();
        }

        #region Pacing & Action Interception

        public override void Enqueue(ISumoAction action)
        {
            if (action == null)
                return;

            pacingController?.RegisterAction(action);
            actionHistory.RegisterOriginalAction(action);

            // Check if action should be filtered
            bool shouldFilter = false;
            if (pacingProfile != null && pacingController != null)
            {
                shouldFilter = pacingProfile.ShouldFilterAction(action, currentPacingFrame.overall);
            }

            if (shouldFilter)
            {
                actionHistory.RegisterFilteredAction(action);
                Debug.Log($"[{ID}] Filtered action: {action.Type} (pacing={currentPacingFrame.overall:F3}, threshold={pacingProfile.filterThreshold:F3})");
            }
            else
            {
                base.Enqueue(action);
            }
        }

        private void InterceptAndFilterActions()
        {
            // Actions are already filtered in Enqueue() override
            // This method is a placeholder for potential future per-frame filtering logic
        }

        private PacingProfile LoadPacingProfile()
        {
            if (wrappedBot == null)
            {
                Debug.LogWarning("[PacedAgent] No wrapped bot assigned!");
                return null;
            }

            string botID = wrappedBot.ID;
            string botSuffix = ExtractBotSuffix(botID);
            string resourcePath = $"Pacing/Pacing_{botSuffix}";

            PacingProfile profile = Resources.Load<PacingProfile>(resourcePath);
            if (profile != null)
            {
                Debug.Log($"[{botID}] Loaded pacing profile: {resourcePath}");
            }
            else
            {
                Debug.Log($"[{botID}] Pacing profile not found at: Assets/Resources/{resourcePath}.asset");
            }

            return profile;
        }

        private string ExtractBotSuffix(string botID)
        {
            // Handle common patterns: "Bot_BT" -> "BT", "AIBot_DQN" -> "DQN"
            if (botID.Contains("_"))
            {
                return botID.Split('_')[^1];
            }
            return botID;
        }

        private void LogActionSummary()
        {
            Debug.Log($"\n========== PACING SUMMARY for {ID} ==========");
            Debug.Log($"Total Original Actions: {actionHistory.OriginalActionCount}");
            Debug.Log($"Total Filtered Actions: {actionHistory.FilteredActionCount}");
            Debug.Log($"Filtering Rate: {(actionHistory.FilteringRate * 100f):F1}%");

            if (actionHistory.OriginalActionCount > 0)
            {
                Debug.Log("\n--- Original Action Breakdown ---");
                foreach (var kvp in actionHistory.ActionTypeCount.OrderByDescending(x => x.Value))
                {
                    Debug.Log($"  {kvp.Key}: {kvp.Value}");
                }
            }

            if (actionHistory.FilteredActionCount > 0)
            {
                Debug.Log("\n--- Filtered Action Breakdown ---");
                foreach (var kvp in actionHistory.FilteredActionTypeCount.OrderByDescending(x => x.Value))
                {
                    Debug.Log($"  {kvp.Key}: {kvp.Value}");
                }
            }

            if (pacingController != null && pacingController.History.Count > 0)
            {
                var history = pacingController.History;
                Debug.Log("\n--- Pacing Statistics ---");
                Debug.Log($"Pacing Segments: {history.Count}");
                Debug.Log($"Avg Threat: {history.Average(f => f.threat):F3}");
                Debug.Log($"Avg Tempo: {history.Average(f => f.tempo):F3}");
                Debug.Log($"Avg Current Pacing: {history.Average(f => f.overall):F3}");
                Debug.Log($"Avg Target Pacing: {history.Average(f => f.target):F3}");
            }

            Debug.Log("=======================================\n");
        }

        #endregion

        #region Inspector Visualization

        public AnimationCurve GetPacingCurve()
        {
            return pacingController?.RuntimeCurve ?? new AnimationCurve();
        }

        public List<PacingFrame> GetPacingHistory()
        {
            return pacingController?.History?.ToList() ?? new List<PacingFrame>();
        }

        public ActionHistory GetActionHistory()
        {
            return actionHistory;
        }

        #endregion
    }

    /// <summary>
    /// Tracks original vs filtered actions for logging and analysis.
    /// </summary>
    public class ActionHistory
    {
        public int OriginalActionCount { get; private set; }
        public int FilteredActionCount { get; private set; }
        public float FilteringRate => OriginalActionCount > 0 ? (float)FilteredActionCount / OriginalActionCount : 0f;
        public Dictionary<ActionType, int> ActionTypeCount { get; } = new();
        public Dictionary<ActionType, int> FilteredActionTypeCount { get; } = new();

        public void Reset()
        {
            OriginalActionCount = 0;
            FilteredActionCount = 0;
            ActionTypeCount.Clear();
            FilteredActionTypeCount.Clear();
        }

        public void RegisterOriginalAction(ISumoAction action)
        {
            if (action == null)
                return;

            OriginalActionCount++;
            if (!ActionTypeCount.ContainsKey(action.Type))
                ActionTypeCount[action.Type] = 0;
            ActionTypeCount[action.Type]++;
        }

        public void RegisterFilteredAction(ISumoAction action)
        {
            if (action == null)
                return;

            FilteredActionCount++;
            if (!FilteredActionTypeCount.ContainsKey(action.Type))
                FilteredActionTypeCount[action.Type] = 0;
            FilteredActionTypeCount[action.Type]++;
        }
    }
}
