using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using CoreSumo;
using UnityEditor;
using UnityEngine;

namespace BotAI
{
    public class EA_MCTS_Node
    {
        public string ID;
        public EA_MCTS_Node head;
        public EA_MCTS_Node parent;
        public List<EA_MCTS_Node> children = new();
        public int visits = 0;
        public float totalReward = 0f;
        public float angleScore = 0f;
        public float distScore = 0f;
        public float bonusOrPenalty = 0f;
        public List<ISumoAction> actions = new();
        public string actionString;
        public List<ISumoAction> badAction;
        public List<ISumoAction> goodAction;
        public bool badActionAlreadyUsed = false;
        public bool goodActionAlreadyUsed = false;
        public PlayerSide Side;

        public HighestScoreType GetHighestScoreType()
        {
            try
            {
                Dictionary<float, HighestScoreType> highestScore = new()
                {
                    {angleScore,HighestScoreType.Angle},
                    {distScore,HighestScoreType.Distance},
                    {bonusOrPenalty,HighestScoreType.BonusOrPenalty}
                };
                Debug.Log($"GetHighestScoreType {string.Join(", ", highestScore.Select((x) => x.Key.ToString()))}");

                float result = highestScore.Max((i) => i.Key);
                return highestScore[result];
            }
            catch (Exception)
            {

                return HighestScoreType.Random;
            }
        }

        public EA_MCTS_Node(EA_MCTS_Node parent, List<ISumoAction> actions, List<ISumoAction> goodAction = null, List<ISumoAction> badAction = null)
        {
            this.parent = parent;
            this.actions = actions;
            this.badAction = badAction;
            this.goodAction = goodAction;
        }

        public void Init(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            children.Clear();
            totalReward = 0f;
            visits = 0;
            foreach (var action in actions)
            {
                EA_MCTS_Node newNode = new(this, new List<ISumoAction>() { action });
                newNode.ID = action.NameWithParam;
                if (goodAction != null && goodAction.Count > 0)
                {
                    newNode.totalReward = 10;
                    newNode.visits = 1;
                }
                if (badAction != null && badAction.Count > 0)
                {
                    newNode.totalReward = -10;
                    newNode.visits = 1;
                }
                children.Add(newNode);
                AllNodes.Add(action.NameWithParam, newNode);
            }
        }

        public EA_MCTS_Node Expand(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            var unexploredActs = AIBot_EA_MCTS.PossibleActions.Where(x =>
            {
                var newActNames = $"{ID}:{x.NameWithParam}";
                if (!AllNodes.ContainsKey(newActNames))
                {
                    return true;
                }
                return false;
            }).ToList();

            if (unexploredActs.Count() == 0)
            {
                return null;
            }

            System.Random random = new();
            var randomAction = unexploredActs[random.Next(unexploredActs.Count())];
            string newActNames;
            newActNames = $"{ID}:{randomAction.NameWithParam}";

            var newActs = new List<ISumoAction>(actions)
                {
                    randomAction
                };

            EA_MCTS_Node newNode = new(this, newActs);
            newNode.ID = newActNames;
            children.Add(newNode);
            AllNodes.Add(newActNames, newNode);
            return newNode;
        }

        public EA_MCTS_Node Select()
        {
            if (children.Count == 0) return this;
            double C = 1.41;
            return children.OrderByDescending(child =>
            {
                if (child.visits == 0) return double.MaxValue;
                double exploitation = child.totalReward / child.visits;
                double exploration = C * Math.Sqrt(Math.Log(visits + 1) / child.visits);
                return exploitation + exploration;
            }).First();
        }

        public Tuple<float, float, float> Simulate(SumoController enemy, SumoController controller, float simulationTime)
        {
            GameObject arena = BattleManager.Instance.Arena;
            float arenaRadius = arena.GetComponent<CircleCollider2D>().radius * arena.transform.lossyScale.x;
            Vector3 arenaCenter = arena.transform.position;
            Vector3 aiDirection = controller.transform.up;
            Vector3 aiPosition = controller.transform.position;

            float bonusOrPenalty = 0;
            float angleScore = 0;
            float distScore = 0;

            List<string> actionsInString = actions.Select((a) => a.Name.ToLower()).ToList();

            bool isActionIncludeAccelerating = actionsInString.Contains("accelerate") || actionsInString.Contains("dash") || actionsInString.Contains("skill");

            foreach (var action in actions)
            {
                if (action is TurnLeftAngleAction)
                {
                    aiDirection += Quaternion.Euler(0, 0, (float)action.Param) * aiDirection * simulationTime * controller.TurnRate;
                }
                else if (action is TurnRightAngleAction rV)
                {
                    aiDirection += Quaternion.Euler(0, 0, (float)action.Param) * aiDirection * simulationTime * controller.TurnRate;
                }
                else if (action is AccelerateAction)
                {
                    if (controller.IsMovementLocked || controller.IsMoveDisabled)
                    {
                        bonusOrPenalty += -0.1f;
                    }
                    else
                    {
                        aiPosition += aiDirection.normalized * controller.MoveSpeed * simulationTime;
                    }
                }
                else if (action is DashAction)
                {
                    if (controller.IsDashCooldown || controller.IsMovementLocked || controller.IsMoveDisabled)
                    {
                        bonusOrPenalty += -0.1f;
                    }
                    else
                    {
                        bonusOrPenalty += 0.1f;
                        aiPosition += aiDirection.normalized * (controller.DashSpeed * controller.DashDuration) * controller.StopDelay * simulationTime;
                    }

                }
                else if (action is SkillAction)
                {
                    if (controller.Skill.IsSkillCooldown)
                    {
                        bonusOrPenalty += -0.5f;
                    }
                    else
                    {
                        if (controller.Skill.Type == ERobotSkillType.Boost)
                        {
                            bonusOrPenalty += 0.5f;
                            aiPosition += aiDirection.normalized * (controller.MoveSpeed * controller.Skill.BoostMultiplier) * simulationTime;
                        }
                    }
                }

                Vector3 toEnemy = enemy.transform.position - aiPosition;
                float distance = toEnemy.magnitude;
                float angle = Vector3.SignedAngle(aiDirection, toEnemy.normalized, Vector3.forward);

                angleScore += Mathf.Cos(angle * Mathf.Deg2Rad);
                distScore += 1f - Mathf.Clamp01(distance / arenaRadius);

                var distanceFromCenter = Vector3.Distance(aiPosition, arenaCenter);

                if ((distanceFromCenter > arenaRadius) && isActionIncludeAccelerating)
                {
                    // Penalize heavily if sumo will exits the ring, or any action that makes the Sumo move away from exits, reward instead.
                    bonusOrPenalty += arenaRadius - distanceFromCenter + (angleScore - 0.9f) * 2;
                    Debug.Log($"[Simulate][IsPossibleOutFromArena] {ID}, can cause go outside of arena\n Detail: {aiPosition}, {arenaCenter} > {arenaRadius}, resulting: {bonusOrPenalty}");
                }
                else
                {
                    distScore += (angleScore > 0.95 && isActionIncludeAccelerating) ? angleScore * 1.5f : 0;
                }
            }

            float normAngleScore = angleScore / actions.Count();
            float normDistScore = distScore / actions.Count();
            float normBonusOrPenalty = bonusOrPenalty / actions.Count();

            this.angleScore += normAngleScore;
            this.distScore += normBonusOrPenalty;
            this.bonusOrPenalty += normBonusOrPenalty;
            return Tuple.Create(normAngleScore, normDistScore, normBonusOrPenalty);
        }

        public void Backpropagate(Tuple<float, float, float> reward)
        {
            visits++;
            totalReward += reward.Item1 + reward.Item2 + reward.Item3;
            angleScore += reward.Item1;
            distScore += reward.Item2;
            bonusOrPenalty += reward.Item3;
            parent?.Backpropagate(reward);
        }

        public EA_MCTS_Node GetBestChild()
        {
            if (children.Count == 0) return null;

            var highest = children.OrderByDescending(child =>
            {
                double exploitation = child.totalReward / (child.visits + double.Epsilon);
                return exploitation;
            }).First();
            return highest;
        }
    }

    public enum HighestScoreType
    {
        Angle,
        Distance,
        BonusOrPenalty,
        Random,
    }
}