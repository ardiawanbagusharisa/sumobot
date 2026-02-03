using System.Collections.Generic;
using UnityEngine;
using SumoCore;

namespace SumoPacing
{
    /// <summary>
    /// Represents target pacing values for a segment
    /// Based on the pacing curve that visualizes pacing per segment
    /// </summary>
    [System.Serializable]
    public class Pacing
    {
        public float threat;  // Target threat level (distance to edge, collision, skill, angle effectiveness)
        public float tempo;   // Target tempo level (action intensity, density, distance, velocity effectiveness)

        public Pacing(float threat = 0f, float tempo = 0f)
        {
            this.threat = threat;
            this.tempo = tempo;
        }
    }

    /// <summary>
    /// PacingController filters and modifies actions based on target pacing patterns
    /// Integration point: SumoController.FlushInput() - between InputProvider.Flush() and Actions.Enqueue()
    /// </summary>
    public class PacingController
    {
        // Target pacing curve/pattern (designer-defined)
        private List<Pacing> targetPacings;

        // Historical actual pacing values: <threat, tempo>
        private List<(float threat, float tempo)> actualPacings;

        // Currently filtered/paced actions
        private Queue<ISumoAction> pacedActions;

        // Reference to the SumoController being paced
        private SumoController sumoController;

        // Current segment index in the pacing curve
        private int currentSegment = 0;

        public PacingController(SumoController controller)
        {
            sumoController = controller;
            targetPacings = new List<Pacing>();
            actualPacings = new List<(float, float)>();
            pacedActions = new Queue<ISumoAction>();
        }
 
        /// <summary>
        /// Set the target pacing curve/pattern
        /// </summary>
        public void SetTargetPacings(List<Pacing> pacings)
        {
            targetPacings = pacings;
        }

        /// <summary>
        /// Main filtering method called from SumoController.FlushInput()
        /// Determines if an action should be enqueued based on target pacing
        /// </summary>
        public bool ShouldEnqueueAction(SumoController controller, ISumoAction action)
        {
            // If no target pacing is set, allow all actions
            if (targetPacings == null || targetPacings.Count == 0)
                return true;

            // Calculate current pacing factors from game state
            PacingFactors factors = CalculatePacingFactors(controller, action);

            // Get current target pacing
            Pacing target = GetCurrentTargetPacing();

            // Evaluate if action matches target pacing
            bool shouldEnqueue = EvaluateAction(factors, target, action);

            return shouldEnqueue;
        }

        /// <summary>
        /// Record action for historical pacing analysis
        /// Called after action is enqueued
        /// </summary>
        public void RecordAction(SumoController controller, ISumoAction action)
        {
            PacingFactors factors = CalculatePacingFactors(controller, action);

            // Calculate actual threat and tempo from factors
            float actualThreat = CalculateThreatFromFactors(factors);
            float actualTempo = CalculateTempoFromFactors(factors);

            actualPacings.Add((actualThreat, actualTempo));
        }

        /// <summary>
        /// Get the current target pacing based on segment
        /// </summary>
        private Pacing GetCurrentTargetPacing()
        {
            if (targetPacings == null || targetPacings.Count == 0)
                return new Pacing(0, 0);

            // Clamp segment index
            int index = Mathf.Clamp(currentSegment, 0, targetPacings.Count - 1);
            return targetPacings[index];
        }

        /// <summary>
        /// Advance to next segment in pacing curve
        /// </summary>
        public void AdvanceSegment()
        {
            currentSegment++;
        }

        /// <summary>
        /// Reset pacing controller state
        /// </summary>
        public void Reset()
        {
            currentSegment = 0;
            actualPacings.Clear();
            pacedActions.Clear();
        }

        #region Pacing Factor Calculations (Placeholder - to be implemented with formulas)

        /// <summary>
        /// Container for all pacing factors from the diagram
        /// </summary>
        private struct PacingFactors
        {
            // Threat factors
            public float distToEdge;
            public float struckCollision;
            public float skillAvail;
            public float enemyAngle;

            // Tempo factors
            public float actionIntensity;
            public float actionDensity;
            public float distToEnemy;
            public float diffVelocity;
        }

        /// <summary>
        /// Calculate all pacing factors from current game state
        /// Based on PacingFactors box in diagram
        /// </summary>
        private PacingFactors CalculatePacingFactors(SumoController controller, ISumoAction action)
        {
            PacingFactors factors = new PacingFactors();

            // TODO: Implement actual calculations based on diagram formulas
            // These are placeholders for now

            // Threat factors
            factors.distToEdge = 0f;          // distance to edge effectiveness
            factors.struckCollision = 0f;     // struck collision effectiveness
            factors.skillAvail = 0f;          // skill availability effectiveness
            factors.enemyAngle = 0f;          // angle to enemy effectiveness

            // Tempo factors
            factors.actionIntensity = 0f;     // action intensity effectiveness
            factors.actionDensity = 0f;       // action density effectiveness
            factors.distToEnemy = 0f;         // distance to enemy effectiveness
            factors.diffVelocity = 0f;        // velocity effectiveness

            return factors;
        }

        /// <summary>
        /// Calculate overall threat value from factors
        /// Formula from diagram (to be implemented)
        /// </summary>
        private float CalculateThreatFromFactors(PacingFactors factors)
        {
            // TODO: Implement formula from diagram
            // threat = f(distToEdge, struckCollision, skillAvail, enemyAngle)
            return 0f;
        }

        /// <summary>
        /// Calculate overall tempo value from factors
        /// Formula from diagram (to be implemented)
        /// </summary>
        private float CalculateTempoFromFactors(PacingFactors factors)
        {
            // TODO: Implement formula from diagram
            // tempo = f(actionIntensity, actionDensity, distToEnemy, diffVelocity)
            return 0f;
        }

        /// <summary>
        /// Evaluate if action should be allowed based on target pacing and constraints
        /// Uses PacingConstraints from diagram (low/up limits)
        /// </summary>
        private bool EvaluateAction(PacingFactors factors, Pacing target, ISumoAction action)
        {
            // TODO: Implement pacing constraints evaluation
            // - Calculate threat/tempo from factors
            // - Normalize using PacingConstraints (min/max/avg/stddev from 13 agents)
            // - Compare with target pacing
            // - Return true if within acceptable range

            // For now, allow all actions (passthrough)
            return true;
        }

        #endregion

        #region Public API for Analysis

        /// <summary>
        /// Get all recorded actual pacing values
        /// Used for calibration and evaluation
        /// </summary>
        public List<(float threat, float tempo)> GetActualPacings()
        {
            return new List<(float, float)>(actualPacings);
        }

        /// <summary>
        /// Get current segment index
        /// </summary>
        public int GetCurrentSegment()
        {
            return currentSegment;
        }

        #endregion
    }

}