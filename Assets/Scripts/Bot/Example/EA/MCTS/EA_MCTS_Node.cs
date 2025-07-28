using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using SumoCore;
using UnityEngine;

namespace SumoBot.EA.MCTS
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
                    newNode.totalReward = 1;
                    newNode.visits = 1;
                }
                if (badAction != null && badAction.Count > 0)
                {
                    newNode.totalReward = -1;
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

        public void SortAction()
        {
            actions.Sort((x, y) =>
            {
                if (x.FullName.ToLower().Contains("turn"))
                {
                    return -100;
                }
                return x.FullName.GetHashCode().CompareTo(y.FullName.GetHashCode());
            });
        }

        public EA_MCTS_Node Select(AI_MCTS_Config config)
        {
            if (children.Count == 0) return this;
            float C = config.UCBConstant;

            return children.OrderByDescending(child =>
            {
                if (child.visits == 0) return double.MaxValue;
                float exploitation = child.totalReward / child.visits;
                double exploration = C * Math.Sqrt(Math.Log(visits + 1) / child.visits);
                return exploitation + exploration;
            }).FirstOrDefault();
        }

        public (float, float, float) Simulate(SumoAPI api, AI_MCTS_Config config)
        {

            float bonusOrPenalty = 0;
            float angleScore = 0;
            float distScore = 0;

            foreach (var action in actions)
            {
                Vector2 predRBPos;
                float predRBRot;
                Vector2 predTPos;
                float predTRot;

                (predRBPos, predRBRot) = api.Simulate(action);
                (predTPos, predTRot) = api.Simulate(action);

                bool approachWithPosition = action is AccelerateAction || action is DashAction;
                float preAngleScore = api.Angle(oriPos: predRBPos, oriRot: predRBRot, normalized: true);
                float preDistScore = api.DistanceNormalized(oriPos: predRBPos);

                angleScore += preAngleScore;
                distScore += preDistScore;

                if (action.Type == ActionType.SkillStone)
                {
                    if (api.CanExecute(action))
                        bonusOrPenalty += 10f;
                    else
                        bonusOrPenalty -= 0.1f;
                    // continue;
                }
                else if (action.Type == ActionType.SkillBoost)
                {
                    if (api.CanExecute(action))
                        bonusOrPenalty += 10f;
                    else
                        bonusOrPenalty -= 0.1f;
                    // continue;
                }
                else if (action.Type == ActionType.Dash)
                {
                    if (!api.CanExecute(action))
                        bonusOrPenalty -= 0.1f;
                    // continue;
                }

                if (action is TurnAction)
                {
                    if (api.Angle(normalized: true) > preAngleScore)
                    {
                        bonusOrPenalty -= 1f;
                    }
                    else if (preAngleScore > 0.9f)
                    {
                        bonusOrPenalty += 2f;
                    }
                }
                else if (action is AccelerateAction)
                {

                    if (preAngleScore > 0.85f)
                    {
                        bonusOrPenalty += 3f;
                    }
                }
                else if (action is DashAction && !api.MyRobot.IsDashOnCooldown)
                {
                    if (preAngleScore > 0.95f)
                    {
                        bonusOrPenalty += 5f;
                    }
                }

                float angleToArena = api.Angle(targetPos: api.BattleInfo.ArenaPosition, oriPos: predTPos);
                Vector2 distanceFromArena = api.Distance(targetPos: api.BattleInfo.ArenaPosition, oriPos: predTPos);

                bonusOrPenalty += (((api.BattleInfo.ArenaRadius * 0.9f) - distanceFromArena.magnitude) * 2) + ((0.5f - preAngleScore) * 2);
            }

            float normAngleScore = angleScore / actions.Count();
            float normDistScore = distScore / actions.Count();
            float normBonusOrPenalty = bonusOrPenalty / actions.Count();

            this.angleScore += normAngleScore;
            this.distScore += normBonusOrPenalty;
            this.bonusOrPenalty += normBonusOrPenalty;
            return (normAngleScore, normDistScore, normBonusOrPenalty);
        }

        public void Backpropagate((float, float, float) reward)
        {
            visits++;
            totalReward += reward.Item1 + reward.Item2 + reward.Item3;
            angleScore += reward.Item1;
            distScore += reward.Item2;
            bonusOrPenalty += reward.Item3;
            parent?.Backpropagate(reward);
        }

        public EA_MCTS_Node GetBestChild(List<EA_MCTS_Node> nodes = null)
        {
            if (children.Count == 0) return null;

            var highest = children.OrderByDescending(child =>
            {
                double exploitation = child.totalReward / (child.visits + double.Epsilon);
                return exploitation;
            }).FirstOrDefault();

            // return highest;

            if (nodes == null)
            {
                nodes = new() { highest };
            }
            else
            {
                nodes.Add(highest);
            }

            if (highest.children.Count == 0)
            {
                if (nodes == null)
                    return highest;
                else
                {
                    return nodes.OrderByDescending(child =>
                    {
                        double exploitation = child.totalReward / (child.visits + double.Epsilon);
                        return exploitation;
                    }).FirstOrDefault();
                }
            }
            return highest.GetBestChild(nodes);
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