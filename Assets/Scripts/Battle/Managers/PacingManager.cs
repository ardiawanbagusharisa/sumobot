using PacingFramework;
using SumoBot;
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

		[Header("Left Player Pacing Configuration")]
		[Tooltip("Fallback pacing filename for left player (human). Can be overridden by Bot.PacingFileName")]
		public string LeftPacingFileName = "Default";
		public float LeftSegmentDuration = 2f;
		public int LeftCollisionWindowSize = 2;

		[Header("Right Player Pacing Configuration")]
		[Tooltip("Fallback pacing filename for right player (human). Can be overridden by Bot.PacingFileName")]
		public string RightPacingFileName = "Default";
		public float RightSegmentDuration = 2f;
		public int RightCollisionWindowSize = 2;

		#endregion

		#region Runtime Properties

		public PacingHandler LeftPacingHandler { get; private set; }
		public PacingHandler RightPacingHandler { get; private set; }

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

		void OnDestroy()
		{
			// Cleanup handlers
			LeftPacingHandler?.Dispose();
			RightPacingHandler?.Dispose();
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
				// Cleanup existing handler
				LeftPacingHandler?.Dispose();

				string finalPacingFileName = LeftPacingFileName;

				if (string.IsNullOrEmpty(finalPacingFileName))
				{
					Logger.Warning($"[PacingManager][Initialize][{controller.Side}] LeftPacingFileName is empty, using Default.json");
					finalPacingFileName = "Default";
				}

				// Create new handler
				LeftPacingHandler = new PacingHandler(
					controller,
					finalPacingFileName,
					LeftSegmentDuration,
					LeftCollisionWindowSize
				);

				// Initialize
				LeftPacingHandler.Init();

				Debug.Log($"[PacingManager] Left handler initialized with PacingFile='{finalPacingFileName}', SegmentDuration={LeftSegmentDuration}s, WindowSize={LeftCollisionWindowSize}");
			}
			else
			{
				RightPacingHandler?.Dispose();

				string finalPacingFileName = RightPacingFileName;
				if (string.IsNullOrEmpty(finalPacingFileName))
				{
					Logger.Warning($"[PacingManager][Initialize][{controller.Side}] LeftPacingFileName is empty, using Default.json");
					finalPacingFileName = "Default";
				}

				// Create new handler
				RightPacingHandler = new PacingHandler(
					controller,
					finalPacingFileName,
					RightSegmentDuration,
					RightCollisionWindowSize
				);

				// Initialize
				RightPacingHandler.Init();

				Debug.Log($"[PacingManager] Right handler initialized with PacingFile='{finalPacingFileName}', SegmentDuration={RightSegmentDuration}s, WindowSize={RightCollisionWindowSize}");
			}
		}

		/// <summary>
		/// Resolves which pacing filename to use based on priority:
		/// 1. If managerFileName is filled and botFileName is provided and not empty -> use botFileName (Bot overrides)
		/// 2. If managerFileName is filled and botFileName is empty/null -> use managerFileName (Manager default for human)
		/// 3. If managerFileName is empty and botFileName is provided -> use botFileName (Bot's config)
		/// 4. Otherwise -> use empty string (will use default constraints)
		/// </summary>
		private string ResolvePacingFileName(string managerFileName, string botFileName)
		{
			// If manager has a filename set
			if (!string.IsNullOrEmpty(managerFileName))
			{
				// If bot also has a filename, bot overrides
				if (!string.IsNullOrEmpty(botFileName))
				{
					Debug.Log($"[PacingManager] Bot PacingFileName '{botFileName}' overrides manager's '{managerFileName}'");
					return botFileName;
				}
				// Otherwise use manager's filename (for human players)
				return managerFileName;
			}

			// Manager has no filename, use bot's (or empty if bot also has none)
			return botFileName ?? "Default";
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
