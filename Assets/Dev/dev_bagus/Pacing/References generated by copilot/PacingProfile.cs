using System;
using System.Collections.Generic;
using System.Linq;
using SumoCore;
using SumoInput;
using UnityEngine;

namespace SumoBot
{
    public enum PacingPattern
    {
        Custom,
        ConstantLow,
        ConstantBalanced,
        ConstantHigh,
        LinearIncrease,
        LinearDecrease,
        ExponentialIncrease,
        ExponentialDecrease,
    }

    [Serializable]
    public struct MinMax
    {
        public float min;
        public float max;

        public MinMax(float min, float max)
        {
            this.min = min;
            this.max = max;
        }
    }

    [Serializable]
    public class PacingConstraints
    {
        [Header("Threat")]
        public MinMax collisionRatio = new(0f, 1f);
        public MinMax enemySkill = new(0f, 1f);
        public MinMax enemyAngle = new(180f, 0f);
        public MinMax agentAngle = new(180f, 0f);
        public MinMax agentDistEdge = new(6f, 0f);
        public MinMax enemyDistEdge = new(6f, 0f);

        [Header("Tempo")]
        public MinMax actionCount = new(0f, 10f);
        public MinMax actionVariation = new(0f, 6f);
        public MinMax avgDistanceToEnemy = new(8f, 0f);
        public MinMax agentVelocity = new(0f, 8f);
        public MinMax enemyVelocity = new(0f, 8f);

        public static PacingConstraints DefaultGlobal()
        {
            return new PacingConstraints();
        }

        public PacingConstraints Clone()
        {
            return new PacingConstraints
            {
                collisionRatio = collisionRatio,
                enemySkill = enemySkill,
                enemyAngle = enemyAngle,
                agentAngle = agentAngle,
                agentDistEdge = agentDistEdge,
                enemyDistEdge = enemyDistEdge,
                actionCount = actionCount,
                actionVariation = actionVariation,
                avgDistanceToEnemy = avgDistanceToEnemy,
                agentVelocity = agentVelocity,
                enemyVelocity = enemyVelocity,
            };
        }
    }

    [Serializable]
    public class PacingFactorWeights
    {
        [Header("Threat weights")]
        public float collision = 1f;
        public float enemySkill = 1f;
        public float deltaAngle = 1f;
        public float deltaDistance = 1f;

        [Header("Tempo weights")]
        public float actionIntensity = 1f;
        public float actionDensity = 1f;
        public float avgDistanceToEnemy = 1f;
        public float deltaVelocity = 1f;

        public static PacingFactorWeights Default()
        {
            return new PacingFactorWeights();
        }
    }

    [Serializable]
    public class SegmentConstraint
    {
        public int segmentIndex = 0;
        [Range(0f, 1f)] public float weight = 1f;
        public PacingConstraints constraints = PacingConstraints.DefaultGlobal();
    }

    [CreateAssetMenu(menuName = "SumoBot/PacingProfile", fileName = "PacingProfile")]
    public class PacingProfile : ScriptableObject
    {
        [Header("Timing")]
        [Range(1f, 60f)] public float segmentDuration = 10f;
        [Tooltip("Used to normalize the target pacing curve. If 0, battle duration will be used at runtime.")]
        public float referenceDuration = 60f;

        [Header("Weights")]
        [Range(0f, 1f)] public float threatWeight = 0.5f;
        [Range(0f, 1f)] public float tempoWeight = 0.5f;
        public PacingFactorWeights factorWeights = PacingFactorWeights.Default();

        [Header("Constraints")]
        public PacingConstraints globalConstraints = PacingConstraints.DefaultGlobal();
        [Tooltip("Optional per-segment constraints. Indices are zero-based.")]
        public List<SegmentConstraint> localConstraints = new();

        [Header("Target pacing")]
        public PacingPattern pattern = PacingPattern.Custom;
        public AnimationCurve customCurve = AnimationCurve.Linear(0f, 0.5f, 1f, 0.5f);
        [Range(0f, 1f)] public float constantLow = 0.2f;
        [Range(0f, 1f)] public float constantBalanced = 0.5f;
        [Range(0f, 1f)] public float constantHigh = 0.8f;
        [Range(0.1f, 4f)] public float exponentialK = 2f;

        [Header("Local softness")]
        [Tooltip("How strongly local constraints override globals (0=ignore locals, 1=fully apply).")]
        [Range(0f, 1f)] public float localConstraintBlend = 1f;

        [Header("Action Filtering")]
        [Tooltip("Enable action filtering based on pacing value.")]
        public bool enableActionFiltering = false;

        [Tooltip("Action types to filter (filter when pacing is below threshold).")]
        public List<ActionType> filterableActions = new();

        [Tooltip("Pacing threshold below which filterableActions are suppressed (0=always allow, 1=always filter).")]
        [Range(0f, 1f)] public float filterThreshold = 0.5f;

        public float EvaluateTarget(float elapsed, float battleDuration)
        {
            float duration = referenceDuration > 0f ? referenceDuration : Mathf.Max(1f, battleDuration);
            float t = Mathf.Clamp01(elapsed / duration);

            return pattern switch
            {
                PacingPattern.ConstantLow => constantLow,
                PacingPattern.ConstantBalanced => constantBalanced,
                PacingPattern.ConstantHigh => constantHigh,
                PacingPattern.LinearIncrease => t,
                PacingPattern.LinearDecrease => 1f - t,
                PacingPattern.ExponentialIncrease => Mathf.Pow(t, exponentialK),
                PacingPattern.ExponentialDecrease => 1f - Mathf.Pow(t, exponentialK),
                _ => Mathf.Clamp01(customCurve.Evaluate(t)),
            };
        }

        public bool ShouldFilterAction(ISumoAction action, float pacingValue)
        {
            if (!enableActionFiltering || action == null)
                return false;

            if (!filterableActions.Contains(action.Type))
                return false;

            return pacingValue < filterThreshold;
        }
    }

    public readonly struct PacingFactors
    {
        public readonly float collision;
        public readonly float enemySkill;
        public readonly float deltaAngle;
        public readonly float deltaDistance;
        public readonly float actionIntensity;
        public readonly float actionDensity;
        public readonly float avgDistanceToEnemy;
        public readonly float deltaVelocity;

        public PacingFactors(
            float collision,
            float enemySkill,
            float deltaAngle,
            float deltaDistance,
            float actionIntensity,
            float actionDensity,
            float avgDistanceToEnemy,
            float deltaVelocity)
        {
            this.collision = collision;
            this.enemySkill = enemySkill;
            this.deltaAngle = deltaAngle;
            this.deltaDistance = deltaDistance;
            this.actionIntensity = actionIntensity;
            this.actionDensity = actionDensity;
            this.avgDistanceToEnemy = avgDistanceToEnemy;
            this.deltaVelocity = deltaVelocity;
        }
    }

    public readonly struct PacingFrame
    {
        public readonly int segmentIndex;
        public readonly float elapsed;
        public readonly float threat;
        public readonly float tempo;
        public readonly float overall;
        public readonly float target;
        public readonly PacingFactors factors;

        public PacingFrame(
            int segmentIndex,
            float elapsed,
            float threat,
            float tempo,
            float overall,
            float target,
            PacingFactors factors)
        {
            this.segmentIndex = segmentIndex;
            this.elapsed = elapsed;
            this.threat = threat;
            this.tempo = tempo;
            this.overall = overall;
            this.target = target;
            this.factors = factors;
        }
    }

    /// <summary>
    /// Runtime pacing controller. Collects per-segment telemetry, normalizes against constraints, and outputs pacing scores.
    /// Can be dropped into any bot and fed with actions/collisions/samples.
    /// </summary>
    public class PacingController
    {
        private readonly PacingProfile profile;
        private readonly SegmentAccumulator accumulator = new();
        private readonly List<PacingFrame> history = new();
        private AnimationCurve runtimeCurve = new();

        private SumoAPI api;
        private int currentSegment = -1;
        private float battleDuration;
        private PacingFrame currentFrame;

        public IReadOnlyList<PacingFrame> History => history;
        public AnimationCurve RuntimeCurve => runtimeCurve;
        public PacingFrame CurrentFrame => currentFrame;

        public PacingController(PacingProfile profile)
        {
            this.profile = profile;
        }

        public void Init(SumoAPI api)
        {
            this.api = api;
            battleDuration = api.BattleInfo.Duration;
            currentSegment = -1;
            accumulator.Reset();
            history.Clear();
            runtimeCurve = new AnimationCurve();
        }

        public void RegisterAction(ISumoAction action)
        {
            accumulator.RegisterAction(action);
        }

        public void RegisterCollision(BounceEvent bounce)
        {
            accumulator.RegisterCollision(bounce, api);
        }

        public PacingFrame Tick()
        {
            if (profile == null || api == null)
            {
                currentFrame = default;
                return currentFrame;
            }

            float elapsed = Mathf.Clamp(api.BattleInfo.Duration - api.BattleInfo.TimeLeft, 0f, battleDuration);
            int segmentIndex = Mathf.FloorToInt(elapsed / profile.segmentDuration);

            if (segmentIndex != currentSegment)
            {
                FinalizeSegment(elapsed);
                accumulator.Reset();
                currentSegment = segmentIndex;
            }

            accumulator.Sample(api);

            PacingConstraints constraints = ResolveConstraints(segmentIndex);
            PacingFactors factors = ComputeFactors(accumulator, api, constraints);

            float threat = WeightedAverage(
                new[] { factors.collision, factors.enemySkill, factors.deltaAngle, factors.deltaDistance },
                new[] { profile.factorWeights.collision, profile.factorWeights.enemySkill, profile.factorWeights.deltaAngle, profile.factorWeights.deltaDistance });

            float tempo = WeightedAverage(
                new[] { factors.actionIntensity, factors.actionDensity, factors.avgDistanceToEnemy, factors.deltaVelocity },
                new[] { profile.factorWeights.actionIntensity, profile.factorWeights.actionDensity, profile.factorWeights.avgDistanceToEnemy, profile.factorWeights.deltaVelocity });

            float overall = CombineThreatTempo(threat, tempo);
            float target = profile.EvaluateTarget(elapsed, battleDuration);

            currentFrame = new PacingFrame(segmentIndex, elapsed, threat, tempo, overall, target, factors);

            if (history.Count == 0 || history[^1].segmentIndex != segmentIndex)
            {
                history.Add(currentFrame);
                runtimeCurve.AddKey(elapsed, overall);
            }
            else
            {
                history[^1] = currentFrame;
                runtimeCurve.MoveKey(runtimeCurve.length - 1, new Keyframe(elapsed, overall));
            }

            return currentFrame;
        }

        private void FinalizeSegment(float elapsed)
        {
            if (history.Count == 0 && currentSegment < 0) return;
        }

        private PacingConstraints ResolveConstraints(int segmentIndex)
        {
            PacingConstraints constraints = profile.globalConstraints.Clone();
            SegmentConstraint local = profile.localConstraints.FirstOrDefault(c => c.segmentIndex == segmentIndex);
            if (local != null)
            {
                float weight = Mathf.Clamp01(local.weight * profile.localConstraintBlend);
                constraints = BlendConstraints(constraints, local.constraints, weight);
            }
            return constraints;
        }

        private static PacingConstraints BlendConstraints(PacingConstraints global, PacingConstraints local, float weight)
        {
            PacingConstraints blended = global.Clone();
            if (Mathf.Approximately(weight, 0f)) return blended;

            blended.collisionRatio = Lerp(global.collisionRatio, local.collisionRatio, weight);
            blended.enemySkill = Lerp(global.enemySkill, local.enemySkill, weight);
            blended.enemyAngle = Lerp(global.enemyAngle, local.enemyAngle, weight);
            blended.agentAngle = Lerp(global.agentAngle, local.agentAngle, weight);
            blended.agentDistEdge = Lerp(global.agentDistEdge, local.agentDistEdge, weight);
            blended.enemyDistEdge = Lerp(global.enemyDistEdge, local.enemyDistEdge, weight);
            blended.actionCount = Lerp(global.actionCount, local.actionCount, weight);
            blended.actionVariation = Lerp(global.actionVariation, local.actionVariation, weight);
            blended.avgDistanceToEnemy = Lerp(global.avgDistanceToEnemy, local.avgDistanceToEnemy, weight);
            blended.agentVelocity = Lerp(global.agentVelocity, local.agentVelocity, weight);
            blended.enemyVelocity = Lerp(global.enemyVelocity, local.enemyVelocity, weight);

            return blended;
        }

        private static MinMax Lerp(MinMax a, MinMax b, float t)
        {
            return new MinMax
            {
                min = Mathf.Lerp(a.min, b.min, t),
                max = Mathf.Lerp(a.max, b.max, t)
            };
        }

        private static float CombineThreatTempo(float threat, float tempo)
        {
            float wThreat = Mathf.Max(0.0001f, Mathf.Abs(threat));
            float wTempo = Mathf.Max(0.0001f, Mathf.Abs(tempo));
            return (threat * wThreat + tempo * wTempo) / (wThreat + wTempo);
        }

        private static float WeightedAverage(IReadOnlyList<float> values, IReadOnlyList<float> weights)
        {
            float total = 0f;
            float weightSum = 0f;
            for (int i = 0; i < values.Count; i++)
            {
                float w = Mathf.Max(0f, weights[i]);
                weightSum += w;
                total += values[i] * w;
            }
            return weightSum > 0f ? total / weightSum : 0f;
        }

        private static float Normalize(float value, MinMax range)
        {
            float denom = range.max - range.min;
            if (Mathf.Approximately(denom, 0f)) return 0.5f;
            float normalized = (value - range.min) / denom;
            return Mathf.Clamp01(normalized);
        }

        private static float DistanceToEdge(Vector2 position, BattleInfoAPI battleInfo)
        {
            float distToCenter = Vector2.Distance(position, battleInfo.ArenaPosition);
            return Mathf.Max(0f, battleInfo.ArenaRadius - distToCenter);
        }

        private static float FacingDelta(float myAverageAngle, float enemyAverageAngle)
        {
            float enemyFacingMe = Mathf.InverseLerp(180f, 0f, enemyAverageAngle);
            float iAmBehindEnemy = 1f - Mathf.InverseLerp(180f, 0f, myAverageAngle);
            return Mathf.Clamp01(enemyFacingMe * iAmBehindEnemy);
        }

        private static PacingFactors ComputeFactors(SegmentAccumulator acc, SumoAPI api, PacingConstraints constraints)
        {
            float collisionRatio = acc.TotalCollisions > 0 ? (float)acc.StruckCollisions / acc.TotalCollisions : 0f;
            float collision = Normalize(collisionRatio, constraints.collisionRatio);

            float skillState = acc.Samples > 0 ? acc.EnemySkillSum / acc.Samples : (api.EnemyRobot.Skill.IsActive ? 1f : api.EnemyRobot.Skill.IsSkillOnCooldown ? 0f : 0.5f);
            float enemySkill = Normalize(skillState, constraints.enemySkill);

            float avgAgentAngle = acc.Samples > 0 ? acc.AgentAngleSum / acc.Samples : Mathf.Abs(api.Angle());
            float avgEnemyAngle = acc.Samples > 0 ? acc.EnemyAngleSum / acc.Samples : Mathf.Abs(api.Angle(oriPos: api.EnemyRobot.Position, oriRot: api.EnemyRobot.Rotation, targetPos: api.MyRobot.Position));
            float deltaAngle = Normalize(FacingDelta(avgAgentAngle, avgEnemyAngle), new MinMax(0f, 1f));

            float avgAgentEdge = acc.Samples > 0 ? acc.AgentEdgeSum / acc.Samples : DistanceToEdge(api.MyRobot.Position, api.BattleInfo);
            float avgEnemyEdge = acc.Samples > 0 ? acc.EnemyEdgeSum / acc.Samples : DistanceToEdge(api.EnemyRobot.Position, api.BattleInfo);
            float edgeDelta = avgEnemyEdge - avgAgentEdge; // positive when we are closer to edge
            float edgeRange = Mathf.Max(constraints.agentDistEdge.min, constraints.enemyDistEdge.min);
            edgeRange = Mathf.Max(edgeRange, 0.1f);
            float deltaDistance = Normalize(edgeDelta, new MinMax(-edgeRange, edgeRange));

            float actionIntensity = Normalize(acc.ActionCount, constraints.actionCount);

            int possibleActions = Enum.GetValues(typeof(ActionType)).Length;
            float actionDensityRaw = possibleActions > 0 ? (float)acc.UniqueActionTypes.Count / possibleActions : 0f;
            float actionDensity = Normalize(actionDensityRaw * constraints.actionVariation.max, constraints.actionVariation);

            float avgDist = acc.Samples > 0 ? acc.DistanceToEnemySum / acc.Samples : Vector2.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
            float avgDistanceToEnemy = Normalize(avgDist, constraints.avgDistanceToEnemy);

            float avgAgentVel = acc.Samples > 0 ? acc.AgentVelocitySum / acc.Samples : api.MyRobot.LinearVelocity.magnitude;
            float avgEnemyVel = acc.Samples > 0 ? acc.EnemyVelocitySum / acc.Samples : api.EnemyRobot.LinearVelocity.magnitude;
            float velocityDelta = avgAgentVel - avgEnemyVel;
            float deltaVelocity = Normalize(velocityDelta, new MinMax(-constraints.enemyVelocity.max, constraints.agentVelocity.max));

            return new PacingFactors(
                collision,
                enemySkill,
                deltaAngle,
                deltaDistance,
                actionIntensity,
                actionDensity,
                avgDistanceToEnemy,
                deltaVelocity);
        }

        private class SegmentAccumulator
        {
            public int ActionCount { get; private set; }
            public HashSet<ActionType> UniqueActionTypes { get; } = new();
            public int TotalCollisions { get; private set; }
            public int StruckCollisions { get; private set; }

            public float DistanceToEnemySum { get; private set; }
            public float AgentEdgeSum { get; private set; }
            public float EnemyEdgeSum { get; private set; }
            public float AgentAngleSum { get; private set; }
            public float EnemyAngleSum { get; private set; }
            public float AgentVelocitySum { get; private set; }
            public float EnemyVelocitySum { get; private set; }
            public float EnemySkillSum { get; private set; }
            public int Samples { get; private set; }

            public void Reset()
            {
                ActionCount = 0;
                UniqueActionTypes.Clear();
                TotalCollisions = 0;
                StruckCollisions = 0;
                DistanceToEnemySum = 0f;
                AgentEdgeSum = 0f;
                EnemyEdgeSum = 0f;
                AgentAngleSum = 0f;
                EnemyAngleSum = 0f;
                AgentVelocitySum = 0f;
                EnemyVelocitySum = 0f;
                EnemySkillSum = 0f;
                Samples = 0;
            }

            public void RegisterAction(ISumoAction action)
            {
                if (action == null) return;
                ActionCount++;
                UniqueActionTypes.Add(action.Type);
            }

            public void RegisterCollision(BounceEvent bounce, SumoAPI api)
            {
                if (bounce == null) return;
                TotalCollisions++;
                if (api != null && bounce.Actor == api.MyRobot.Side)
                    StruckCollisions++;
            }

            public void Sample(SumoAPI api)
            {
                if (api == null) return;
                Samples++;

                DistanceToEnemySum += Vector2.Distance(api.MyRobot.Position, api.EnemyRobot.Position);
                AgentEdgeSum += DistanceToEdge(api.MyRobot.Position, api.BattleInfo);
                EnemyEdgeSum += DistanceToEdge(api.EnemyRobot.Position, api.BattleInfo);
                AgentAngleSum += Mathf.Abs(api.Angle());
                EnemyAngleSum += Mathf.Abs(api.Angle(
                    oriPos: api.EnemyRobot.Position,
                    oriRot: api.EnemyRobot.Rotation,
                    targetPos: api.MyRobot.Position));
                AgentVelocitySum += api.MyRobot.LinearVelocity.magnitude;
                EnemyVelocitySum += api.EnemyRobot.LinearVelocity.magnitude;
                EnemySkillSum += api.EnemyRobot.Skill.IsActive
                    ? 1f
                    : api.EnemyRobot.Skill.IsSkillOnCooldown
                        ? 0f
                        : 0.5f;
            }
        }
    }
}
