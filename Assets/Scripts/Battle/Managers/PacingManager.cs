using PacingFramework;
using SumoCore;
using UnityEngine;
namespace SumoManager
{
	/// <summary>
	/// Manages pacing systems for both left and right players.
	/// Responsible for initializing, configuring, and coordinating pacing handlers.
	/// Provides comparison and aggregate statistics between left and right pacing.
	/// </summary>
	public class PacingManager : MonoBehaviour
	{
		public static PacingManager Instance { get; private set; }

		#region Inspector Configuration

		[Tooltip("Minimum pacing value for percentile mapping (0th percentile)")]
		public float MinPacing = 0.0f;

		[Tooltip("Maximum pacing value for percentile mapping (100th percentile)")]
		public float MaxPacing = 0.43f;

		public bool RandomTarget = false;

		[Header("Left Player Pacing Configuration")]
		[Tooltip("Fallback pacing filename for left player (human). Can be overridden by Bot.PacingFileName")]
		public string LeftFileName = "Default";
		public float LeftSegmentDuration = 2f;
		[Tooltip("Time-based collision window duration in seconds (e.g., 3 seconds lookback)")]
		public float LeftCollisionWindowDuration = 3f;
		public bool LeftActionFiltering = true;
		public bool LeftNNCandidates = true;
		public bool LeftMCTSCandidates = false;


		[Header("Right Player Pacing Configuration")]
		[Tooltip("Fallback pacing filename for right player (human). Can be overridden by Bot.PacingFileName")]
		public string RightFileName = "Default";
		public float RightSegmentDuration = 2f;
		[Tooltip("Time-based collision window duration in seconds (e.g., 3 seconds lookback)")]
		public float RightCollisionWindowDuration = 3f;

		public bool RightActionFiltering = true;
		public bool RightNNCandidates = true;
		public bool RightMCTSCandidates = false;

		#endregion

		#region Pacing Simulation Overrides (set programmatically by BattleSimulator)

		// When both Left*SimPath fields are non-empty, Initialize() builds the Left handler's
		// PacingTarget from these two files (merged) instead of LeftFileName. Cleared automatically
		// when Pacing Simulation is not active. Not meant to be hand-edited in the inspector.
		[HideInInspector] public string LeftSimTargetPath;
		[HideInInspector] public string LeftSimConstraintPath;
		[HideInInspector] public string RightSimTargetPath;
		[HideInInspector] public string RightSimConstraintPath;

		#endregion

		#region Runtime Properties

		public PacingHandler LeftPacingHandler { get; private set; }
		public PacingHandler RightPacingHandler { get; private set; }

		// Persistent pacing histories that survive rematch/Battle_Start
		private GamePacing leftPacingHistory = new GamePacing();
		private GamePacing rightPacingHistory = new GamePacing();

		private PacingBrainHeuristic leftPacingBrainHeuristic = null;
		private PacingBrainHeuristic rightPacingBrainHeuristic = null;

		#endregion

		#region Unity Lifecycle

		void Awake()
		{
			if (Instance != null)
			{
				Destroy(gameObject);
				return;
			}
			Instance = this;
		}

		void OnEnable()
		{
			BattleManager.Instance.Events[BattleManager.OnBattleChanged].Subscribe(OnBattleChanged);
		}

		void OnDisable()
		{
			BattleManager.Instance.Events[BattleManager.OnBattleChanged].Unsubscribe(OnBattleChanged);
		}

		void OnDestroy()
		{
			// Cleanup handlers
			LeftPacingHandler?.Dispose();
			RightPacingHandler?.Dispose();

			// Save PacingBrain models before destruction (NN only - heuristic doesn't save)
			leftPacingBrainHeuristic?.SaveModelToDisk();  // No-op
			rightPacingBrainHeuristic?.SaveModelToDisk();  // No-op
		}

		void Update()
		{
			if (LeftPacingHandler != null && LeftPacingHandler.EnableActionFiltering != LeftActionFiltering)
			{
				LeftPacingHandler.EnableActionFiltering = LeftActionFiltering;
			}

			if (RightPacingHandler != null && RightPacingHandler.EnableActionFiltering != RightActionFiltering)
			{
				RightPacingHandler.EnableActionFiltering = RightActionFiltering;
			}
		}

		#endregion

		#region Initialization Methods

		/// <summary>
		/// Initialize left pacing handler.
		/// </summary>
		/// <param name="side">The PlayerSide for the left/right player</param>
		/// <param name="controller">The SumoController for the player</param>
		/// <param name="botPacingFileName">Optional: Bot's pacing filename (from Bot.PacingFileName). If provided and not empty, overrides PacingFileName.</param>
		public void Initialize(PlayerSide side, SumoController controller)
		{
			if (!enabled) return;

			if (side == PlayerSide.Left)
			{
				// Cleanup existing handler (but keep PacingBrain alive!)
				LeftPacingHandler?.Dispose();

				string finalPacingFileName = LeftFileName;

				if (string.IsNullOrEmpty(finalPacingFileName))
				{
					Logger.Warning($"[PacingManager][Initialize][{controller.Side}] LeftPacingFileName is empty, using Default.json");
					finalPacingFileName = "Default";
				}

				if (leftPacingBrainHeuristic == null)
				{
					leftPacingBrainHeuristic = new PacingBrainHeuristic(controller);
					Debug.Log($"[PacingManager] Created new Left PacingBrain Heuristic instance (no training required)");
				}

				LeftPacingHandler = new PacingHandler(
					controller,
					finalPacingFileName,
					LeftSegmentDuration,
					LeftCollisionWindowDuration,
					leftPacingHistory,
					MinPacing,
					MaxPacing,
					leftPacingBrainHeuristic,  // Pass persistent heuristic brain (may be null)
					LeftNNCandidates,
					LeftMCTSCandidates,
					LeftSimTargetPath,
					LeftSimConstraintPath
				);

				// Set the direct reference on controller for action filtering
				controller.PacingHandler = LeftPacingHandler;

				// Initialize
				LeftPacingHandler.Init();

				Debug.Log($"[PacingManager] Left handler initialized with PacingFile='{LeftSimTargetPath ?? finalPacingFileName}', SegmentDuration={LeftSegmentDuration}s, WindowDuration={LeftCollisionWindowDuration}s");
			}
			else
			{
				// Cleanup existing handler (but keep PacingBrain alive!)
				RightPacingHandler?.Dispose();

				string finalPacingFileName = RightFileName;
				if (string.IsNullOrEmpty(finalPacingFileName))
				{
					Logger.Warning($"[PacingManager][Initialize][{controller.Side}] RightPacingFileName is empty, using Default.json");
					finalPacingFileName = "Default";
				}

				if (rightPacingBrainHeuristic == null)
				{
					rightPacingBrainHeuristic = new PacingBrainHeuristic(controller);
					Debug.Log($"[PacingManager] Created new Right PacingBrain Heuristic instance (no training required)");
				}

				RightPacingHandler = new PacingHandler(
					controller,
					finalPacingFileName,
					RightSegmentDuration,
					RightCollisionWindowDuration,
					rightPacingHistory,
					MinPacing,
					MaxPacing,
					rightPacingBrainHeuristic,  // Pass persistent heuristic brain (may be null)
					RightNNCandidates,
					RightMCTSCandidates,
					RightSimTargetPath,
					RightSimConstraintPath
				);

				// Set the direct reference on controller for action filtering
				controller.PacingHandler = RightPacingHandler;

				// Initialize
				RightPacingHandler.Init();

				Debug.Log($"[PacingManager] Right handler initialized with PacingFile='{RightSimTargetPath ?? finalPacingFileName}', SegmentDuration={RightSegmentDuration}s, WindowDuration={RightCollisionWindowDuration}s");
			}
		}

		#endregion

		#region Round Management

		/// <summary>
		/// Initialize pacing history for a new round. Should be called at the start of each round.
		/// Randomizes target pacing values for more natural training.
		/// </summary>
		public void InitRound()
		{
			if (!enabled) return;
			leftPacingHistory.InitBattle();
			rightPacingHistory.InitBattle();

			// Randomize pacing targets for natural training variation
			if (RandomTarget)
				RandomizePacingTargets();

			Debug.Log("[PacingManager] Round pacing history initialized with randomized targets");
		}

		/// <summary>
		/// Randomizes the threat and tempo targets for both handlers.
		/// Each target value is randomized within a reasonable range around 0.5 (±0.3).
		/// This creates natural variation for training while keeping targets achievable.
		/// </summary>
		private void RandomizePacingTargets()
		{
			// Randomize left handler targets
			if (LeftPacingHandler != null && LeftPacingHandler.PacingTarget != null)
			{
				RandomizeTargetList(LeftPacingHandler.PacingTarget.ThreatTargets);
				RandomizeTargetList(LeftPacingHandler.PacingTarget.TempoTargets);
				Debug.Log($"[PacingManager] Left targets randomized - Sample: Threat[0]={LeftPacingHandler.PacingTarget.ThreatTargets[0]:F3}, Tempo[0]={LeftPacingHandler.PacingTarget.TempoTargets[0]:F3}");
			}

			// Randomize right handler targets
			if (RightPacingHandler != null && RightPacingHandler.PacingTarget != null)
			{
				RandomizeTargetList(RightPacingHandler.PacingTarget.ThreatTargets);
				RandomizeTargetList(RightPacingHandler.PacingTarget.TempoTargets);
				Debug.Log($"[PacingManager] Right targets randomized - Sample: Threat[0]={RightPacingHandler.PacingTarget.ThreatTargets[0]:F3}, Tempo[0]={RightPacingHandler.PacingTarget.TempoTargets[0]:F3}");
			}
		}

		/// <summary>
		/// Randomizes all values in a target list.
		/// </summary>
		private void RandomizeTargetList(System.Collections.Generic.List<float> targets)
		{
			const float MIN_TARGET = 0.2f;
			const float MAX_TARGET = 0.8f;

			for (int i = 0; i < targets.Count; i++)
			{
				targets[i] = Random.Range(MIN_TARGET, MAX_TARGET);
			}
		}

		public void OnBattleChanged(EventParameter param)
		{
			if (param.BattleState == BattleState.Battle_Ongoing)
			{
				InitRound();
			}
		}

		#endregion

		#region Tick Methods

		/// <summary>
		/// Tick both pacing handlers. Should be called from BattleManager's action interval.
		/// </summary>
		public void Tick()
		{
			if (!enabled) return;
			LeftPacingHandler?.Tick();
			RightPacingHandler?.Tick();
		}

		#endregion

		#region Comparison & Statistics Methods

		/// <summary>
		/// Compare current pacing between left and right players.
		/// Returns a comparison summary with threat and tempo differences.
		/// </summary>
		public PacingComparison GetCurrentComparison()
		{
			if (LeftPacingHandler == null || RightPacingHandler == null)
				return null;

			var leftPacing = LeftPacingHandler.GetCurrentSegmentPacing();
			var rightPacing = RightPacingHandler.GetCurrentSegmentPacing();

			if (leftPacing == null || rightPacing == null)
				return null;

			return new PacingComparison
			{
				LeftThreat = leftPacing.Threat.Value,
				RightThreat = rightPacing.Threat.Value,
				ThreatDifference = leftPacing.Threat.Value - rightPacing.Threat.Value,
				LeftTempo = leftPacing.Tempo.Value,
				RightTempo = rightPacing.Tempo.Value,
				TempoDifference = leftPacing.Tempo.Value - rightPacing.Tempo.Value,
				LeftOverall = leftPacing.GetOverallPacing(),
				RightOverall = rightPacing.GetOverallPacing(),
				OverallDifference = leftPacing.GetOverallPacing() - rightPacing.GetOverallPacing()
			};
		}

		/// <summary>
		/// Get aggregate statistics from both handlers.
		/// Calculates averages across all completed segments in current round.
		/// </summary>
		public PacingAggregateStats GetAggregateStats()
		{
			if (LeftPacingHandler == null || RightPacingHandler == null)
				return null;

			var leftHistory = LeftPacingHandler.GetHistory();
			var rightHistory = RightPacingHandler.GetHistory();

			var leftRound = leftHistory.CurrentRound();
			var rightRound = rightHistory.CurrentRound();

			if (leftRound.SegmentPacings.Count == 0 && rightRound.SegmentPacings.Count == 0)
				return null;

			return new PacingAggregateStats
			{
				LeftAverageThreat = CalculateAverageThreat(leftRound),
				RightAverageThreat = CalculateAverageThreat(rightRound),
				LeftAverageTempo = CalculateAverageTempo(leftRound),
				RightAverageTempo = CalculateAverageTempo(rightRound),
				LeftSegmentCount = leftRound.SegmentPacings.Count,
				RightSegmentCount = rightRound.SegmentPacings.Count,
				MoreAggressiveSide = DetermineMoreAggressive(leftRound, rightRound),
				MoreActiveSide = DetermineMoreActive(leftRound, rightRound)
			};
		}

		private float CalculateAverageThreat(GamePacingItem round)
		{
			if (round.SegmentPacings.Count == 0)
				return 0f;

			float sum = 0f;
			foreach (var segment in round.SegmentPacings)
			{
				sum += segment.Threat.Value;
			}
			return sum / round.SegmentPacings.Count;
		}

		private float CalculateAverageTempo(GamePacingItem round)
		{
			if (round.SegmentPacings.Count == 0)
				return 0f;

			float sum = 0f;
			foreach (var segment in round.SegmentPacings)
			{
				sum += segment.Tempo.Value;
			}
			return sum / round.SegmentPacings.Count;
		}

		private PlayerSide DetermineMoreAggressive(GamePacingItem leftRound, GamePacingItem rightRound)
		{
			float leftAvgThreat = CalculateAverageThreat(leftRound);
			float rightAvgThreat = CalculateAverageThreat(rightRound);

			return leftAvgThreat > rightAvgThreat ? PlayerSide.Left : PlayerSide.Right;
		}

		private PlayerSide DetermineMoreActive(GamePacingItem leftRound, GamePacingItem rightRound)
		{
			float leftAvgTempo = CalculateAverageTempo(leftRound);
			float rightAvgTempo = CalculateAverageTempo(rightRound);

			return leftAvgTempo > rightAvgTempo ? PlayerSide.Left : PlayerSide.Right;
		}

		/// <summary>
		/// Log current comparison to console.
		/// </summary>
		public void LogCurrentComparison()
		{
			var comparison = GetCurrentComparison();
			if (comparison == null)
			{
				Debug.Log("[PacingManager] No current comparison available");
				return;
			}

			Debug.Log($"[PacingManager] CURRENT COMPARISON\n" +
				$"Threat: Left={comparison.LeftThreat:F3}, Right={comparison.RightThreat:F3}, Diff={comparison.ThreatDifference:F3}\n" +
				$"Tempo: Left={comparison.LeftTempo:F3}, Right={comparison.RightTempo:F3}, Diff={comparison.TempoDifference:F3}\n" +
				$"Overall: Left={comparison.LeftOverall:F3}, Right={comparison.RightOverall:F3}, Diff={comparison.OverallDifference:F3}");
		}

		/// <summary>
		/// Log aggregate statistics to console.
		/// </summary>
		public void LogAggregateStats()
		{
			var stats = GetAggregateStats();
			if (stats == null)
			{
				Debug.Log("[PacingManager] No aggregate stats available");
				return;
			}

			Debug.Log($"[PacingManager] AGGREGATE STATISTICS\n" +
				$"Average Threat: Left={stats.LeftAverageThreat:F3}, Right={stats.RightAverageThreat:F3}\n" +
				$"Average Tempo: Left={stats.LeftAverageTempo:F3}, Right={stats.RightAverageTempo:F3}\n" +
				$"Segment Counts: Left={stats.LeftSegmentCount}, Right={stats.RightSegmentCount}\n" +
				$"More Aggressive: {stats.MoreAggressiveSide}\n" +
				$"More Active: {stats.MoreActiveSide}");
		}

		#endregion
	}

	#region Data Classes

	/// <summary>
	/// Comparison data between left and right pacing for a single segment.
	/// </summary>
	public class PacingComparison
	{
		public float LeftThreat;
		public float RightThreat;
		public float ThreatDifference; // Left - Right (positive = left more threatening)

		public float LeftTempo;
		public float RightTempo;
		public float TempoDifference; // Left - Right (positive = left faster tempo)

		public float LeftOverall;
		public float RightOverall;
		public float OverallDifference; // Left - Right

		public override string ToString()
		{
			return $"Threat: L={LeftThreat:F3} R={RightThreat:F3} Diff={ThreatDifference:F3}, " +
				   $"Tempo: L={LeftTempo:F3} R={RightTempo:F3} Diff={TempoDifference:F3}, " +
				   $"Overall: L={LeftOverall:F3} R={RightOverall:F3} Diff={OverallDifference:F3}";
		}
	}

	/// <summary>
	/// Aggregate pacing statistics across all segments in current round.
	/// </summary>
	public class PacingAggregateStats
	{
		public float LeftAverageThreat;
		public float RightAverageThreat;

		public float LeftAverageTempo;
		public float RightAverageTempo;

		public int LeftSegmentCount;
		public int RightSegmentCount;

		public PlayerSide MoreAggressiveSide; // Higher average threat
		public PlayerSide MoreActiveSide; // Higher average tempo

		public override string ToString()
		{
			return $"AvgThreat: L={LeftAverageThreat:F3} R={RightAverageThreat:F3}, " +
				   $"AvgTempo: L={LeftAverageTempo:F3} R={RightAverageTempo:F3}, " +
				   $"Segments: L={LeftSegmentCount} R={RightSegmentCount}, " +
				   $"MoreAggressive={MoreAggressiveSide}, MoreActive={MoreActiveSide}";
		}
	}

	#endregion
}
