using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using SumoCore;
using SumoManager;
using UnityEditor;
using UnityEngine;

namespace SumoBot
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
                newNode.ID = action.FullName;
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
                AllNodes.Add(action.FullName, newNode);
            }
        }

        public EA_MCTS_Node Expand(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            var unexploredActs = AIBot_EA_MCTS.PossibleActions.Where(x =>
            {
                var newActNames = $"{ID}:{x.FullName}";
                if (!AllNodes.ContainsKey(newActNames))
                    return true;
                return false;
            }).ToList();

            if (unexploredActs.Count() == 0)
                return null;

            System.Random random = new();
            var randomAction = unexploredActs[random.Next(unexploredActs.Count())];
            string newActNames;
            newActNames = $"{ID}:{randomAction.FullName}";

            var newActs = new List<ISumoAction>(actions) { randomAction };

            EA_MCTS_Node newNode = new(this, newActs)
            {
                ID = newActNames
            };
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
            }).FirstOrDefault();
        }

        public Tuple<float, float, float> Simulate(BotAPI api, float simulationTime)
        {
            var controller = api.Controller;

            GameObject arena = BattleManager.Instance.Arena;
            float arenaRadius = arena.GetComponent<CircleCollider2D>().radius * arena.transform.lossyScale.x;

            Vector3 arenaCenter = arena.transform.position;
            Vector3 aiDirection = api.MyTransform.up;
            Vector3 aiPosition = api.MyTransform.position;

            float bonusOrPenalty = 0;
            float angleScore = 0;
            float distScore = 0;

            //Before Sim
            Vector3 toEnemy = api.EnemyTransform.position - aiPosition;
            float distance = toEnemy.magnitude;
            float angle = Vector3.SignedAngle(aiDirection, toEnemy.normalized, Vector3.forward);

            List<string> actionsInString = actions.Select((a) => a.Name.ToLower()).ToList();

            Debug.Log($"distanceScore before-loop: {distScore}, {string.Join(", ", actionsInString)}");

            bool isActionIncludeAccelerating = actionsInString.Contains("accelerate") || actionsInString.Contains("dash") || actionsInString.Contains("skill");


            foreach (var action in actions)
            {
                if (action is TurnAction)
                {
                    Debug.Log($"action.Param ${action.Type} {action.Param}");
                    if (action.Type == ActionType.TurnLeftWithAngle)
                    {
                        aiDirection += Quaternion.Euler(0, 0, (float)action.Param) * aiDirection * simulationTime * controller.TurnRate;
                    }
                    else if (action.Type == ActionType.TurnRightWithAngle)
                    {
                        aiDirection += Quaternion.Euler(0, 0, -(float)action.Param) * aiDirection * simulationTime * controller.TurnRate;
                    }
                }
                else if (action is AccelerateAction)
                {
                    if (api.CanExecute(action))
                    {
                        var predictionSpeed = controller.MoveSpeed;
                        if (controller.Skill.Type == SkillType.Boost && controller.Skill.IsActive)
                        {
                            predictionSpeed *= controller.Skill.BoostMultiplier;
                        }

                        aiPosition += aiDirection.normalized * (predictionSpeed * simulationTime);
                    }
                    else
                    {
                        bonusOrPenalty -= 0.1f;
                    }
                }
                else if (action is DashAction)
                {
                    if (api.CanExecute(action))
                    {
                        bonusOrPenalty += 0.1f;
                        var predictionSpeed = controller.DashSpeed;

                        if (controller.Skill.Type == SkillType.Boost && controller.Skill.IsActive)
                        {
                            predictionSpeed *= controller.Skill.BoostMultiplier;
                        }

                        aiPosition += aiDirection.normalized * (controller.DashDuration * predictionSpeed * simulationTime);

                        // Formula of decelerating / stop-delay
                        aiPosition *= 0.5f + predictionSpeed * controller.StopDelay;
                    }
                    else
                    {
                        bonusOrPenalty -= 0.1f;
                    }

                }
                else if (action is SkillAction)
                {
                    if (api.CanExecute(action))
                    {
                        if (controller.Skill.Type == SkillType.Boost)
                        {
                            bonusOrPenalty += 0.5f;
                            aiPosition += aiDirection.normalized * (controller.MoveSpeed * controller.Skill.BoostMultiplier * simulationTime);
                        }
                    }
                    else
                    {
                        bonusOrPenalty -= 0.5f;
                    }
                }

                toEnemy = api.EnemyTransform.transform.position - aiPosition;
                distance = toEnemy.magnitude;
                angle = Vector3.SignedAngle(aiDirection, toEnemy.normalized, Vector3.forward);

                angleScore += Mathf.Cos(angle * Mathf.Deg2Rad);
                distScore += 1f - Mathf.Clamp01(distance / arenaRadius);

                Debug.Log($"distanceScore after-loop: {distScore}");
                var distanceFromCenter = Vector3.Distance(aiPosition, arenaCenter);

                if (isActionIncludeAccelerating)
                {
                    if (distanceFromCenter > arenaRadius)
                    {
                        // Penalize heavily if sumo will exits the ring, or any action that makes the Sumo move away from exits, reward instead.
                        bonusOrPenalty += (arenaRadius - distanceFromCenter + (angleScore - 0.9f)) * 2;
                        Debug.Log($"[Simulate][IsPossibleOutFromArena] {ID}, can cause go outside of arena\n Detail: {aiPosition}, {arenaCenter} > {arenaRadius}, resulting: {bonusOrPenalty}");
                    }
                    else
                    {
                        distScore += (angleScore > 0.95) ? angleScore * 2f : 0;
                    }
                }
                else
                {
                    if (angleScore > 0.9)
                    {
                        bonusOrPenalty -= 5f;
                    }
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
            }).FirstOrDefault();
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